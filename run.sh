#python train.py --obj capsule

for obj in bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper
do
      python train.py --obj "$obj"
done