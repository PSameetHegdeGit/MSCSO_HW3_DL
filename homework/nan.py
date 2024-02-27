import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# a simple convolutional layer
conv_layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)


input_tensor = torch.zeros(1, 1, 16, 16)

# Set a single element to NaN
input_tensor[0, 0, 4, 5] = float('nan')
#print(f"{input_tensor.data=} before conv operation\n")
featuremaps=[]
# Perform the convolution
output_tensor = conv_layer(input_tensor)
#print(f"{output_tensor.data=} after 1 conv operation\n")
featuremaps.append(output_tensor)
output_tensor = conv_layer(output_tensor)
#print(f"{output_tensor=} after 2 conv operation\n")
featuremaps.append(output_tensor)
output_tensor = conv_layer(output_tensor)
#print(f"{output_tensor=} after 3 conv operation\n")
featuremaps.append(output_tensor)

plt.figure(figsize=(10, 4))
for i,feature_map in enumerate(featuremaps):
    feature_map=feature_map.detach().numpy()
    plt.subplot(1,3, i+1)
    plt.imshow(feature_map[0,0,:,:])
plt.tight_layout()
plt.show()