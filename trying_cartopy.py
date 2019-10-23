# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 14:57:56 2019

@author: kkrao
"""

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.offsetbox import AnchoredText
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature


def main():
    fig = plt.figure()

            
    proj = ccrs.LambertConformal(central_longitude=-95, central_latitude=38, \
                                  false_easting=-0.0, false_northing=0.0, \
                                  standard_parallels=(33,45), \
                                  globe=None)    
 
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    # ax.set_xlim(-170,170)
    # ax.set_ylim(-80,80)
    ax.set_extent([-119, -92, 22.8, 52],crs = ccrs.PlateCarree())

    # Put a background image on for nice sea rendering.
    # ax.stock_img()

    # Create a feature for States/Admin 1 regions at 1:50m from Natural Earth
    # states_provinces = cfeature.NaturalEarthFeature(
    #     category='cultural',
    #     name='admin_1_states_provinces_lines',
    #     scale='50m',
    #     facecolor='none')

    SOURCE = 'Natural Earth'
    LICENSE = 'public domain'

    # ax.add_feature(cfeature.LAND)
    # ax.add_feature(cfeature.COASTLINE)
    fname = r'D:/Krishna/projects/vwc_from_radar/data/usa_shapefile/west_usa/cb_2017_us_state_500k.shp'

    shape_feature = ShapelyFeature(Reader(fname).geometries(),
                                ccrs.PlateCarree(), edgecolor='black')
    ax.add_feature(shape_feature,facecolor = "None")
    # ax.add_feature(cfeature.)
    # ax.add_feature(states_provinces, edgecolor='gray')

    # Add a text annotation for the license information to the
    # the bottom right corner.
    # text = AnchoredText(r'$\mathcircled{{c}}$ {}; license: {}'
                        # ''.format(SOURCE, LICENSE),
                        # loc=4, prop={'size': 12}, frameon=True)
    # ax.add_artist(text)

    plt.show()


if __name__ == '__main__':
    main()