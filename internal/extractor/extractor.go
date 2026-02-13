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

// ExtractPage uses the CGO bridge into libvmi to read a single 4 KiB page.
func ExtractPage(vmName string, address uint64) ([]byte, error) {
	if vmName == "" {
		return nil, fmt.Errorf("vm name is required")
	}

	buf := make([]byte, PageSize)
	if len(buf) != PageSize {
		return nil, fmt.Errorf("failed to allocate %d-byte buffer", PageSize)
	}

	cname := C.CString(vmName)
	defer C.free(unsafe.Pointer(cname))

	res := C.fetch_guest_page(cname, C.uint64_t(address), (*C.uchar)(unsafe.Pointer(&buf[0])))
	if res != 0 {
		return nil, fmt.Errorf("fetch_guest_page returned %d", int(res))
	}

	return buf, nil
}
