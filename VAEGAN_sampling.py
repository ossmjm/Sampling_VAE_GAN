#!/usr/bin/python3

import sys, getopt, os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import make_network as make_network
import utils as utils

print('TF version = ', tf.__version__)

def main(argv):
    file = ''
    try:
        opts, args = getopt.getopt(argv,"hi:",["help","ifile="])
    except getopt.GetoptError:
        print ('Usage: VAEGAN_sample.py -i <inputimage>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print ('Usage: VAEGAN_sample.py -i <inputimage>')
            print ('   Loads vaegan_celeba.ckpt, encodes the image to latent space,')
            print ('   samples 10 variations (anchor + random interpolation), decodes, and saves as JPG.')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            file = arg
            print ('Input file is "', file)

    # Build network
    sess, X, G, Z, Z_mu, is_training, saver = make_network.make_network()
    if os.path.exists("vaegan_celeba.ckpt"):
        saver.restore(sess, "vaegan_celeba.ckpt")
        print("VAE-GAN model restored.")
    else:
        print("Pre-trained network appears to be missing.")
        sys.exit()

    # Load and preprocess input image
    img = plt.imread(file)[..., :3]
    img = utils.preprocess128(img, crop_factor=0.8)[np.newaxis]

    # Encode to latent mean
    z_anchor = sess.run(Z_mu, feed_dict={X: img, is_training: False})

    # Number of output samples
    num_samples = 10
    generated_images = []
    filename = ' '
    for i in range(num_samples):
        # random latent
        z_rand = np.random.normal(0, 1, z_anchor.shape)

        # gradually increasing alpha (0.2, 0.4, ... 2.0)
        alpha = 0.1 * (i+1)
        print(f"alpha={alpha:.2f}")

        # interpolation between anchor and random latent
        z_edit = (1 - alpha) * z_anchor * 0.4  + alpha * z_rand * 0.4

        # Decode to image
        g = sess.run(G, feed_dict={Z: z_edit, is_training: False})

        # Deprocess for saving
        g = np.clip(g[0] / g.max(), 0, 1)
        generated_images.append(g)
        filename = os.path.splitext(os.path.basename(file))[0]
        os.makedirs(filename, exist_ok=True)
        out_file = f"{filename}/{filename}_sample_{i}.jpg"
        plt.imsave(out_file, g)
        print(f"Saved {out_file}")
        
    fig, axes = plt.subplots(1, num_samples, figsize=(2*num_samples, 2))
    for idx, ax in enumerate(axes):
        ax.imshow(generated_images[idx])
        ax.axis('off')
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    collage_file = f"{filename}/{filename}_samples_grid.jpg"
    plt.savefig(collage_file, bbox_inches='tight')
    plt.close()
    print(f"Saved collage grid -> {collage_file}")
if __name__ == "__main__":
    main(sys.argv[1:])
