package extractor

/*
#cgo CFLAGS: -I${SRCDIR}
#include "probe.h"
*/
import "C"

import (
	"fmt"
	"unsafe"
)

const PageSize = int(C.HYPTCN_PAGE_SIZE)

// FetchPage requests a page-sized chunk from the hypothetical hypervisor.
func FetchPage(address uint64) ([]byte, error) {
	buf := make([]byte, PageSize)
	if len(buf) == 0 {
		return nil, fmt.Errorf("failed to allocate buffer")
	}

	res := C.fetch_ram_page(C.uint64_t(address), (*C.char)(unsafe.Pointer(&buf[0])))
	if res != 0 {
		return nil, fmt.Errorf("fetch_ram_page returned %d", int(res))
	}

	return buf, nil
}
