#include "probe.h"
#include <libvmi/libvmi.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

struct hyptcn_vmi_handle {
    vmi_instance_t vmi;
};

hyptcn_vmi_handle_t *hyptcn_vmi_open(const char *vm_name) {
    if (vm_name == NULL) {
        return NULL;
    }

    hyptcn_vmi_handle_t *h = malloc(sizeof(hyptcn_vmi_handle_t));
    if (h == NULL) {
        return NULL;
    }
    h->vmi = NULL;

    /* Use vmi_init (not vmi_init_complete) — we only need raw physical memory
     * reads via vmi_read_pa, so OS-layer init (System.map / rekall profile)
     * is not required. */
    status_t status = vmi_init(
        &h->vmi,
        VMI_KVM,
        vm_name,
        VMI_INIT_DOMAINNAME,
        NULL,
        NULL
    );

    if (status == VMI_FAILURE) {
        fprintf(stderr, "hyptcn: vmi_init failed for \"%s\"\n", vm_name);
        free(h);
        return NULL;
    }

    return h;
}

int hyptcn_read_page(hyptcn_vmi_handle_t *handle,
                     uint64_t physical_address,
                     unsigned char *buffer) {
    if (handle == NULL || buffer == NULL) {
        return -1;
    }

    status_t status = vmi_read_pa(handle->vmi, physical_address,
                                  HYPTCN_PAGE_SIZE, buffer, NULL);
    return (status == VMI_SUCCESS) ? 0 : -2;
}

void hyptcn_vmi_close(hyptcn_vmi_handle_t *handle) {
    if (handle == NULL) {
        return;
    }
    vmi_destroy(handle->vmi);
    free(handle);
}
