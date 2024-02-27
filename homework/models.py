import torch
import torch.nn.functional as F


class CNNClassifier(torch.nn.Module):
    # TODO: Update block if model isn't accurate with layers
    class Block(torch.nn.Module):
        def __init__(self, n_input, n_output, stride=1):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Conv2d(n_input, n_output, kernel_size=3, padding=1, stride=stride),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU(),
            )

        def forward(self, x):
            return self.net(x)

    def __init__(self, layer_channels=(32, 64, 128, 256)):
        super().__init__()
        """
        Your code here
        Hint: Base this on yours or HW2 master solution if you'd like.
        Hint: Overall model can be similar to HW2, but you likely need some architecture changes (e.g. ResNets)
        """
        input_channels = 3
        c = layer_channels[0]

        L = [
            torch.nn.Conv2d(input_channels, c, kernel_size=7, padding=3, stride=2),
            torch.nn.BatchNorm2d(c),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # TODO: Do we need to use a maxpool layer here?
        ]

        if len(layer_channels) > 1:
            for l in layer_channels[1:]:
                L.append(self.Block(c, l, stride=2))
                c = l

        self.network = torch.nn.Sequential(*L)
        self.classifier = torch.nn.Linear(c, 6)

    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        Hint: Apply input normalization inside the network, to make sure it is applied in the grader
        """
        x = self.network(x)
        x = x.mean([2, 3])  # what are these dimensions?
        x = self.classifier(x.view(x.size(0), -1))

        return x



# encoding block
class encoding_block(torch.nn.Module):
    """
    Convolutional batch norm block with relu activation (main block used in the encoding steps)
    """
    def __init__(self, in_size, out_size, kernel_size=3, padding=0, stride=1, dilation=1, batch_norm=True, dropout=False):
        super().__init__()

        if batch_norm:

            # reflection padding for same size output as input (reflection padding has shown better results than zero padding)
            layers = [torch.nn.ReflectionPad2d(padding=(kernel_size -1)//2),
                      torch.nn.Conv2d(in_size, out_size, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation),
                      torch.nn.PReLU(),
                      torch.nn.BatchNorm2d(out_size),
                      torch.nn.ReflectionPad2d(padding=(kernel_size - 1)//2),
                      torch.nn.Conv2d(out_size, out_size, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation),
                      torch.nn.PReLU(),
                      torch.nn.BatchNorm2d(out_size),
                      ]

        else:
            layers = [torch.nn.ReflectionPad2d(padding=(kernel_size - 1)//2),
                      torch.nn.Conv2d(in_size, out_size, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation),
                      torch.nn.PReLU(),
                      torch.nn.ReflectionPad2d(padding=(kernel_size - 1)//2),
                      torch.nn.Conv2d(out_size, out_size, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation),
                      torch.nn.PReLU(),]

        if dropout:
            layers.append(torch.nn.Dropout())

        self.encoding_block = torch.nn.Sequential(*layers)

    def forward(self, input):

        output = self.encoding_block(input)

        return output


# decoding block
class decoding_block(torch.nn.Module):
    def __init__(self, in_size, out_size, batch_norm=False, upsampling=True):
        super().__init__()

        if upsampling:
            self.up = torch.nn.Sequential(torch.nn.Upsample(mode='bilinear', scale_factor=2),
                                    torch.nn.Conv2d(in_size, out_size, kernel_size=1))

        else:
            self.up = torch.nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)

        self.conv = encoding_block(in_size, out_size, batch_norm=batch_norm)

    def forward(self, input1, input2):

        output2 = self.up(input2)

        output1 = torch.nn.functional.upsample(input1, output2.size()[2:], mode='bilinear')

        return self.conv(torch.cat([output1, output2], 1))


class FCN(torch.nn.Module):
    '''
    Following U-net design - create parallel downblocks and upblocks
    based on implementation from https://github.com/milesial/Pytorch-UNet/blob/67bf11b4db4c5f2891bd7e8e7f58bcde8ee2d2db/unet/unet_model.py#L8
    and paper: https://arxiv.org/abs/1505.04597v1

    '''


    def __init__(self, base_channels=32):
        super().__init__()
        """
        Your code here.
        Hint: The FCN can be a bit smaller the the CNNClassifier since you need to run it at a higher resolution
        Hint: Use up-convolutions
        Hint: Use skip connections
        Hint: Use residual connections
        Hint: Always pad by kernel_size / 2, use an odd kernel_size
        """

        # encoding
        self.conv1 = encoding_block(3, 32)
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=2)

        self.conv2 = encoding_block(32, 64)
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=2)

        self.conv3 = encoding_block(64, 128)
        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=2)

        self.conv4 = encoding_block(128, 256)
        self.maxpool4 = torch.nn.MaxPool2d(kernel_size=2)

        # center
        self.center = encoding_block(256, 512)

        # decoding
        self.decode4 = decoding_block(512, 256)
        self.decode3 = decoding_block(256, 128)
        self.decode2 = decoding_block(128, 64)
        self.decode1 = decoding_block(64, 32)

        # final
        self.final = torch.nn.Conv2d(32, 5, kernel_size=1)


    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,H,W))
        @return: torch.Tensor((B,5,H,W))
        Hint: Apply input normalization inside the network, to make sure it is applied in the grader
        Hint: Input and output resolutions need to match, use output_padding in up-convolutions, crop the output
              if required (use z = z[:, :, :H, :W], where H and W are the height and width of a corresponding strided
              convolution
        """
        #print(f"shape to test: {x.shape}")


        # encoding
        conv1 = self.conv1(x)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        # center
        center = self.center(maxpool4)

        # decoding
        decode4 = self.decode4(conv4, center)

        decode3 = self.decode3(conv3, decode4)

        decode2 = self.decode2(conv2, decode3)

        decode1 = self.decode1(conv1, decode2)

        # final
        final =torch.nn.functional.upsample(self.final(decode1), x.size()[2:], mode='bilinear')

        return final


model_factory = {
    'cnn': CNNClassifier,
    'fcn': FCN,
}


def save_model(model):
    from torch import save
    from os import path
    for n, m in model_factory.items():
        if isinstance(model, m):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '%s.th' % n))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(model):
    from torch import load
    from os import path
    r = model_factory[model]()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model), map_location='cpu'))
    return r
