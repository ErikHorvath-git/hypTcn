package extractor

/*
#cgo CFLAGS: -I${SRCDIR}
#cgo LDFLAGS: -lvmi
#include <stdlib.h>
#include "probe.h"
*/
import "C"

import (
	"fmt"
	"unsafe"
)

const PageSize = int(C.HYPTCN_PAGE_SIZE)

// Handle wraps the persistent libvmi connection.
type Handle struct {
	ptr *C.hyptcn_vmi_handle_t
}

// ProcessInfo holds per-process data from the guest task_struct list.
type ProcessInfo struct {
	PID  int32
	Name string
	CR3  uint64 // page-table base (physical address)
}

// ModuleInfo holds per-module data from the guest kernel module list.
type ModuleInfo struct {
	Name        string
	BaseAddress uint64
	Size        uint64
}

// ConnectionInfo holds one TCP socket entry from the guest.
type ConnectionInfo struct {
	LocalIP    uint32
	LocalPort  uint16
	RemoteIP   uint32
	RemotePort uint16
	State      uint8
}

// Open connects to the named KVM guest via libvmi in raw physical-memory mode.
// The caller must call Close() when done, typically via defer.
func Open(vmName string) (*Handle, error) {
	if vmName == "" {
		return nil, fmt.Errorf("vm name is required")
	}
	cname := C.CString(vmName)
	defer C.free(unsafe.Pointer(cname))

	ptr := C.hyptcn_vmi_open(cname)
	if ptr == nil {
		return nil, fmt.Errorf("hyptcn_vmi_open failed for %q", vmName)
	}
	return &Handle{ptr: ptr}, nil
}

// OpenWithSysmap connects with OS-layer introspection using the supplied
// System.map file.  If sysmapPath is empty or vmi_init_complete fails,
// it transparently falls back to raw physical-memory mode (same as Open).
func OpenWithSysmap(vmName, sysmapPath string) (*Handle, error) {
	if vmName == "" {
		return nil, fmt.Errorf("vm name is required")
	}
	cname := C.CString(vmName)
	defer C.free(unsafe.Pointer(cname))

	var csysmap *C.char
	if sysmapPath != "" {
		csysmap = C.CString(sysmapPath)
		defer C.free(unsafe.Pointer(csysmap))
	}

	ptr := C.hyptcn_vmi_open_with_sysmap(cname, csysmap)
	if ptr == nil {
		return nil, fmt.Errorf("hyptcn_vmi_open_with_sysmap failed for %q", vmName)
	}
	return &Handle{ptr: ptr}, nil
}

// HasOSLayer returns true if the handle was opened with vmi_init_complete and
// OS-layer symbol resolution (System.map) is active.
func (h *Handle) HasOSLayer() bool {
	return h.ptr != nil && C.hyptcn_has_os_layer(h.ptr) != 0
}

// ReadPage reads exactly PageSize bytes from physicalAddress into a new buffer.
func (h *Handle) ReadPage(physicalAddress uint64) ([]byte, error) {
	buf := make([]byte, PageSize)
	res := C.hyptcn_read_page(h.ptr, C.uint64_t(physicalAddress),
		(*C.uchar)(unsafe.Pointer(&buf[0])))
	if res != 0 {
		return nil, fmt.Errorf("hyptcn_read_page returned %d", int(res))
	}
	return buf, nil
}

// Close destroys the libvmi connection and frees the handle.
func (h *Handle) Close() {
	if h.ptr != nil {
		C.hyptcn_vmi_close(h.ptr)
		h.ptr = nil
	}
}

// GetProcessList walks the guest Linux task_struct list.
// Returns an empty slice (not an error) when the OS layer is not active.
func (h *Handle) GetProcessList() ([]ProcessInfo, error) {
	if h.ptr == nil {
		return nil, fmt.Errorf("handle is closed")
	}
	procs := make([]C.hyptcn_proc_t, C.HYPTCN_MAX_PROCS)
	n := C.hyptcn_get_process_list(h.ptr, &procs[0], C.HYPTCN_MAX_PROCS)
	if n < 0 {
		return nil, fmt.Errorf("hyptcn_get_process_list returned %d", int(n))
	}
	result := make([]ProcessInfo, int(n))
	for i := 0; i < int(n); i++ {
		result[i] = ProcessInfo{
			PID:  int32(procs[i].pid),
			Name: C.GoString(&procs[i].name[0]),
			CR3:  uint64(procs[i].cr3),
		}
	}
	return result, nil
}

// GetKernelModules walks the guest Linux kernel module linked list.
// Returns an empty slice (not an error) when the OS layer is not active.
func (h *Handle) GetKernelModules() ([]ModuleInfo, error) {
	if h.ptr == nil {
		return nil, fmt.Errorf("handle is closed")
	}
	mods := make([]C.hyptcn_module_t, C.HYPTCN_MAX_MODS)
	n := C.hyptcn_get_kernel_modules(h.ptr, &mods[0], C.HYPTCN_MAX_MODS)
	if n < 0 {
		return nil, fmt.Errorf("hyptcn_get_kernel_modules returned %d", int(n))
	}
	result := make([]ModuleInfo, int(n))
	for i := 0; i < int(n); i++ {
		result[i] = ModuleInfo{
			Name:        C.GoString(&mods[i].name[0]),
			BaseAddress: uint64(mods[i].base_address),
			Size:        uint64(mods[i].size),
		}
	}
	return result, nil
}

// GetNetworkConnections walks the guest TCP established-connection hash table.
// Returns an empty slice (not an error) when the OS layer is not active.
func (h *Handle) GetNetworkConnections() ([]ConnectionInfo, error) {
	if h.ptr == nil {
		return nil, fmt.Errorf("handle is closed")
	}
	conns := make([]C.hyptcn_conn_t, C.HYPTCN_MAX_CONNS)
	n := C.hyptcn_get_network_connections(h.ptr, &conns[0], C.HYPTCN_MAX_CONNS)
	if n < 0 {
		return nil, fmt.Errorf("hyptcn_get_network_connections returned %d", int(n))
	}
	result := make([]ConnectionInfo, int(n))
	for i := 0; i < int(n); i++ {
		result[i] = ConnectionInfo{
			LocalIP:    uint32(conns[i].local_ip),
			LocalPort:  uint16(conns[i].local_port),
			RemoteIP:   uint32(conns[i].remote_ip),
			RemotePort: uint16(conns[i].remote_port),
			State:      uint8(conns[i].state),
		}
	}
	return result, nil
}

// TranslateV2P translates the virtual address vaddr in the address space
// identified by dtb (CR3 physical address) to a physical address.
func (h *Handle) TranslateV2P(dtb, vaddr uint64) (uint64, error) {
	if h.ptr == nil {
		return 0, fmt.Errorf("handle is closed")
	}
	var paddr C.uint64_t
	res := C.hyptcn_translate_v2p(h.ptr,
		C.uint64_t(dtb), C.uint64_t(vaddr), &paddr)
	if res != 0 {
		return 0, fmt.Errorf("hyptcn_translate_v2p returned %d", int(res))
	}
	return uint64(paddr), nil
}
