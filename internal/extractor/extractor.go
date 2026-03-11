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

// Open connects to the named KVM guest via libvmi. The caller must call
// Close() when done, typically via defer.
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
