#!/bin/bash
list="dino006 dino013 light009 light031 light032 light039 ornaments006 ornaments007 rice004 rice006 rice007 rice008 rice009 sofa006 sofa007 sofa009 sofa010 sofa011 suitcase001 suitcase_003 suitcase_004 suitcase_005 suitcase_006 suitcase_007 table009 table017 table019 table020 table023 table026"
for i in $list; do
python train.py -s Data/oo3d/${i} -m Output/oo3d/${i} --depth_ratio 0 --geo_white_background
python extract_mesh_tsdf.py -m Output/oo3d/${i}/total
mv Output/oo3d/${i}/total/point_cloud/iteration_20000 Output/oo3d/${i}/total/point_cloud/iteration_20000_geo
mv Output/oo3d/${i}/total/point_cloud/iteration_20000_ref Output/oo3d/${i}/total/point_cloud/iteration_20000
python render_ref.py -s Data/oo3d/${i} -m Output/oo3d/${i}/total --render_images
done