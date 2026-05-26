# Imaris ↔ syGlass Interoperability Pipeline

Goal is shuttle annotations/segmentations between `.ims` files and syGlass projects in user-friendly ways.
Currently:
syGlass -> Imaris works ok for masks
Imaris -> syGlass works natively for points (counting points in syGlass)

TBD:
testing syglass -> imaris counting points
figuring out Imaris surfaces -> syglass.
  imaris surface decoding to binary is OK but unsure whether it's possible to directly make ROI(s) and mask objects.
