#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "gxx_angle_gen_distro.h"
#include "xxx_Band.h"
#include "xxx_Sensor.h"

int main(int argc, char **argv)
{
    gxx_angle_gen_metadata_TYPE metadata;
    if (argc != 2 || gxx_angle_gen_read_ang(argv[1], &metadata) != SUCCESS)
        return EXIT_FAILURE;
    if (xxx_initialize_sensor_type(metadata.spacecraft_id) == IAS_SENSOR_UNKNOWN)
        return EXIT_FAILURE;

    int band = 0;
    while (band < (int)metadata.num_bands &&
           xxx_get_user_band(metadata.band_metadata[band].band_number) != 8)
        ++band;
    if (band == (int)metadata.num_bands)
        return EXIT_FAILURE;

    printf("grid,%d,%d,%.8f,%.8f,%.8f\n",
           metadata.band_metadata[band].l1t_lines,
           metadata.band_metadata[band].l1t_samps,
           metadata.band_metadata[band].pixel_size,
           metadata.corners.upleft.x, metadata.corners.upleft.y);
    const double degrees = 45.0 / atan(1.0);
    double line, sample;
    int count;
    while ((count = scanf("%lf %lf", &line, &sample)) == 2) {
        double sun[2], view[2];
        int outside_sun, outside_view;
        if (gxx_angle_gen_calculate_angles_rpc(&metadata, line, sample, NULL,
                band, 0.0, 1, GXX_ANGLE_GEN_SOLAR, &outside_sun, sun) != SUCCESS ||
            gxx_angle_gen_calculate_angles_rpc(&metadata, line, sample, NULL,
                band, 0.0, 1, GXX_ANGLE_GEN_SATELLITE, &outside_view, view) != SUCCESS)
            return EXIT_FAILURE;
        printf("%d,%d,%.8f,%.8f,%.8f,%.8f\n", outside_sun, outside_view,
               sun[0] * degrees, sun[1] * degrees, view[0] * degrees, view[1] * degrees);
    }
    gxx_angle_gen_free(&metadata);
    return count == EOF ? EXIT_SUCCESS : EXIT_FAILURE;
}
