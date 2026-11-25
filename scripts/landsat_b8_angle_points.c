/* Evaluate USGS Landsat angle coefficients at requested native B8 pixels. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "ias_angle_gen_distro.h"
#include "ias_const.h"
#include "ias_logging.h"
#include "ias_miscellaneous.h"
#include "ias_satellite_attributes.h"

int main(int argc, char **argv)
{
    IAS_ANGLE_GEN_METADATA metadata;
    if (argc != 2 || ias_log_initialize("Landsat B8 points") != SUCCESS
        || ias_sat_attr_initialize(IAS_L8) != SUCCESS
        || ias_angle_gen_read_ang(argv[1], &metadata) != SUCCESS) {
        return EXIT_FAILURE;
    }

    const int index = ias_sat_attr_convert_band_number_to_index(8);
    if (index < 0 || !metadata.band_present[index]) {
        ias_angle_gen_free(&metadata);
        return EXIT_FAILURE;
    }
    const IAS_ANGLE_GEN_BAND *band = &metadata.band_metadata[index];
    IAS_MISC_LINE_EXTENT *trim = ias_misc_create_output_image_trim_lut(
        band->active_l1t_corner_lines, band->active_l1t_corner_samps,
        band->l1t_lines, band->l1t_samps);
    if (trim == NULL) {
        ias_angle_gen_free(&metadata);
        return EXIT_FAILURE;
    }

    const double height_m = 0.0;
    const double degrees = 180.0 / acos(-1.0);
    int line, sample, scanned, result = EXIT_SUCCESS;
    while ((scanned = scanf("%d %d", &line, &sample)) == 2) {
        if (line < 0 || line >= band->l1t_lines || sample <= trim[line].start_sample
            || sample >= trim[line].end_sample) {
            printf("%d,%d,outside_active_image,,,,\n", line, sample);
            continue;
        }

        double view[2], sun[2];
        int outside_view, outside_sun;
        if (ias_angle_gen_calculate_angles_rpc(&metadata, line, sample, &height_m, index,
                IAS_ANGLE_GEN_SATELLITE, &outside_view, view) != SUCCESS
            || ias_angle_gen_calculate_angles_rpc(&metadata, line, sample, &height_m, index,
                IAS_ANGLE_GEN_SOLAR, &outside_sun, sun) != SUCCESS) {
            result = EXIT_FAILURE;
            break;
        }
        if (outside_view || outside_sun) {
            printf("%d,%d,outside_angle_coverage,,,,\n", line, sample);
            continue;
        }
        printf("%d,%d,resolved,%.9f,%.9f,%.9f,%.9f\n", line, sample,
            degrees * sun[IAS_ANGLE_GEN_AZIMUTH_INDEX], degrees * sun[IAS_ANGLE_GEN_ZENITH_INDEX],
            degrees * view[IAS_ANGLE_GEN_AZIMUTH_INDEX], degrees * view[IAS_ANGLE_GEN_ZENITH_INDEX]);
    }
    free(trim);
    ias_angle_gen_free(&metadata);
    return scanned == EOF ? result : EXIT_FAILURE;
}
