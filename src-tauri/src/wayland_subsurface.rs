//! WebKitGTK owns the window's `wl_surface` (#1060, tauri#9220), so wgpu presents to a
//! subsurface stacked below it, visible through the transparent page.

use std::cell::RefCell;
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicI32, Ordering};

use gtk::glib::object::ObjectType;
use gtk::prelude::*;
use wayland_client::backend::{Backend, ObjectId};
use wayland_client::globals::{GlobalListContents, registry_queue_init};
use wayland_client::protocol::{
    wl_compositor::WlCompositor, wl_region::WlRegion, wl_registry::WlRegistry,
    wl_subcompositor::WlSubcompositor, wl_subsurface::WlSubsurface, wl_surface::WlSurface,
};
use wayland_client::{Connection, Dispatch, Proxy, QueueHandle, delegate_noop};
use wgpu::rwh::{RawDisplayHandle, RawWindowHandle, WaylandDisplayHandle, WaylandWindowHandle};

struct State;

static CHILD: OnceLock<WlSurface> = OnceLock::new();
static BUFFER_SCALE: AtomicI32 = AtomicI32::new(1);

struct Placement {
    window: gtk::Widget,
    webview: gtk::Widget,
    subsurface: WlSubsurface,
    offset: (i32, i32),
}

thread_local! {
    // GTK main thread only.
    static PLACEMENT: RefCell<Option<Placement>> = const { RefCell::new(None) };
}

impl Dispatch<WlRegistry, GlobalListContents> for State {
    fn event(
        _: &mut Self,
        _: &WlRegistry,
        _: <WlRegistry as Proxy>::Event,
        _: &GlobalListContents,
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}

delegate_noop!(State: WlCompositor);
delegate_noop!(State: WlSubcompositor);
delegate_noop!(State: WlSubsurface);
delegate_noop!(State: WlRegion);
delegate_noop!(State: ignore WlSurface);

/// Must run on the GTK main thread. `None` on X11 or on failure (JPEG previews).
pub fn create_surface(
    window: &tauri::WebviewWindow,
    instance: &wgpu::Instance,
) -> Option<wgpu::Surface<'static>> {
    if !gtk::is_initialized_main_thread() {
        return None;
    }
    match try_create_surface(window, instance) {
        Ok(surface) => surface,
        Err(e) => {
            log::warn!(
                "Wayland subsurface setup failed, using JPEG previews: {}",
                e
            );
            None
        }
    }
}

fn try_create_surface(
    window: &tauri::WebviewWindow,
    instance: &wgpu::Instance,
) -> Result<Option<wgpu::Surface<'static>>, String> {
    let gtk_window = window.gtk_window().map_err(|e| e.to_string())?;
    let webview = window.default_vbox().map_err(|e| e.to_string())?;
    let gdk_display = gtk_window.display();
    if gdk_display.type_().name() != "GdkWaylandDisplay" {
        return Ok(None);
    }

    // GTK3 creates the toplevel wl_surface at realize and destroys it on hide.
    gtk_window.realize();
    let gdk_window = gtk_window.window().ok_or("GTK window has no GdkWindow")?;
    let scale = gdk_window.scale_factor();

    let (display_ptr, parent_ptr) = unsafe {
        (
            gdk_wayland_sys::gdk_wayland_display_get_wl_display(gdk_display.as_ptr() as *mut _),
            gdk_wayland_sys::gdk_wayland_window_get_wl_surface(gdk_window.as_ptr() as *mut _),
        )
    };
    let display_ptr = NonNull::new(display_ptr).ok_or("no wl_display")?;
    if parent_ptr.is_null() {
        return Err("no toplevel wl_surface".into());
    }

    let conn = Connection::from_backend(unsafe {
        Backend::from_foreign_display(display_ptr.as_ptr() as *mut _)
    });
    let (globals, queue) = registry_queue_init::<State>(&conn).map_err(|e| e.to_string())?;
    let qh = queue.handle();
    let compositor: WlCompositor = globals.bind(&qh, 1..=4, ()).map_err(|e| e.to_string())?;
    let subcompositor: WlSubcompositor = globals.bind(&qh, 1..=1, ()).map_err(|e| e.to_string())?;

    let parent_id = unsafe { ObjectId::from_ptr(WlSurface::interface(), parent_ptr as *mut _) }
        .map_err(|e| e.to_string())?;
    let parent = WlSurface::from_id(&conn, parent_id).map_err(|e| e.to_string())?;

    let child = compositor.create_surface(&qh, ());
    let subsurface = subcompositor.get_subsurface(&child, &parent, &qh, ());
    subsurface.place_below(&parent);
    subsurface.set_desync();
    let empty = compositor.create_region(&qh, ());
    child.set_input_region(Some(&empty));
    empty.destroy();
    child.set_buffer_scale(scale);
    BUFFER_SCALE.store(scale, Ordering::Relaxed);
    child.commit();
    conn.flush().map_err(|e| e.to_string())?;

    let child_ptr = NonNull::new(child.id().as_ptr() as *mut c_void).ok_or("no child surface")?;
    let surface = unsafe {
        instance.create_surface_unsafe(wgpu::SurfaceTargetUnsafe::RawHandle {
            raw_display_handle: Some(RawDisplayHandle::Wayland(WaylandDisplayHandle::new(
                display_ptr,
            ))),
            raw_window_handle: RawWindowHandle::Wayland(WaylandWindowHandle::new(child_ptr)),
        })
    }
    .map_err(|e| e.to_string())?;

    // The subsurface lives as long as the app; its proxies need the queue to stay alive.
    let _ = CHILD.set(child.clone());
    PLACEMENT.with_borrow_mut(|p| {
        *p = Some(Placement {
            window: gtk_window.upcast(),
            webview: webview.upcast(),
            subsurface,
            offset: (0, 0),
        })
    });
    std::mem::forget((conn, queue, compositor, subcompositor, parent, child));
    log::info!(
        "Created Wayland subsurface for direct GPU display (scale {})",
        scale
    );
    Ok(Some(surface))
}

/// Applies with the next presented frame, whose size must be a multiple of `scale`.
pub fn set_buffer_scale(scale: i32) {
    if let Some(child) = CHILD.get()
        && BUFFER_SCALE.swap(scale, Ordering::Relaxed) != scale
    {
        child.set_buffer_scale(scale);
    }
}

/// Keeps the subsurface at the webview's offset in the window, which the native title bar
/// changes. Callable from any thread; applies with the window's next frame.
pub fn sync_position() {
    gtk::glib::MainContext::default().invoke(|| {
        PLACEMENT.with_borrow_mut(|p| {
            let Some(p) = p else { return };
            if let Some(offset) = p.webview.translate_coordinates(&p.window, 0, 0)
                && offset != p.offset
            {
                p.offset = offset;
                p.subsurface.set_position(offset.0, offset.1);
                p.window.queue_draw();
            }
        })
    });
}
