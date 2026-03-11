#include "probe.h"
#include <libvmi/libvmi.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

struct hyptcn_vmi_handle {
    vmi_instance_t vmi;
};

/*
 * Build a minimal libvmi config string for the named Linux guest.
 * Uses vmi_init_complete with VMI_CONFIG_STRING so no /etc/libvmi.conf
 * is required.
 *
 * The config string format is:
 *   <vm_name> { ostype = "Linux"; }
 */
static char *_build_config(const char *vm_name) {
    /* enough room for: name + ' { ostype = "Linux"; }\0' */
    size_t len = strlen(vm_name) + 64;
    char *buf = malloc(len);
    if (!buf) return NULL;
    snprintf(buf, len, "%s { ostype = \"Linux\"; }", vm_name);
    return buf;
}

hyptcn_vmi_handle_t *hyptcn_vmi_open(const char *vm_name) {
    if (vm_name == NULL) {
        return NULL;
    }

    hyptcn_vmi_handle_t *h = malloc(sizeof(hyptcn_vmi_handle_t));
    if (h == NULL) {
        return NULL;
    }
    h->vmi = NULL;

    char *config = _build_config(vm_name);
    if (!config) {
        free(h);
        return NULL;
    }

    vmi_init_error_t err = VMI_INIT_ERROR_NONE;
    status_t status = vmi_init_complete(
        &h->vmi,
        vm_name,
        VMI_INIT_DOMAINNAME,
        NULL,
        VMI_CONFIG_STRING,
        config,
        &err
    );
    free(config);

    if (status == VMI_FAILURE) {
        fprintf(stderr, "hyptcn: vmi_init_complete failed (error=%d)\n", (int)err);
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
