# Kelvin6K Troubleshooting Guide

## Delamination
Delamination is the separation of adjacent printed layers, where a layer fails to
bond to the one beneath it and a visible split or gap forms along the layer line.
In large-format concrete printing it is usually caused by too long an interval
between layers, so the lower layer has cured too far before the next is placed;
by a dry or poorly mixed material batch; or by low ambient humidity and high
temperature accelerating surface drying. Delaminated walls lose structural
strength and must be inspected before further layers are added.

## Layer Shifting
Layer shifting is a horizontal offset between one layer and the next, so the wall
leans or steps rather than rising vertically. Common causes are mechanical: a
loose drive belt or coupling, a skipped step on a drive motor, an obstruction in
the gantry path, or the printed part being knocked while still green. It can also
be caused by an unstable base layer that slumps under the weight of layers above.
Uneven layer pitch measured by the vision system is an early indicator.

## Poor Layer Bonding
Poor layer bonding happens when adjacent layers touch but do not fuse into a
continuous section. Typical causes are insufficient material flow at the nozzle,
too fast a print speed for the extrusion rate, a layer interval that is too long,
contamination or dust on the previous layer surface, and incorrect accelerator
dosing. Bonding problems often appear first as reduced bead width, so a narrowing
layer width trend should be investigated promptly.

## Over-extrusion and Wide Layers
A layer that is wider than the configured maximum indicates over-extrusion:
too much material for the travel speed. Causes include pump speed set too high,
accelerator dosing too low so the material stays fluid, or the nozzle standing
off too close to the previous layer. Excessively wide layers slump under load,
distort wall geometry, and waste material.

## Under-extrusion and Narrow Layers
A layer narrower than the configured minimum indicates under-extrusion. Causes
include a partially clogged nozzle, low material in the hopper, a failing pump
seal, or air entrained in the delivery line. Narrow layers reduce wall thickness
and weaken the structure.
