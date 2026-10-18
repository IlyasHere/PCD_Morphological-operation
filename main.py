import numpy as np
import matplotlib.pyplot as plt

def dilation(image, se):
    """Dilation menggunakan structuring element 3x3."""
    img_h, img_w = image.shape
    se_h, se_w = se.shape

    pad_h = se_h // 2
    pad_w = se_w // 2

    padded_img = np.pad(
        image,
        ((pad_h, pad_h), (pad_w, pad_w)),
        mode='constant',
        constant_values=0
    )

    output = np.zeros_like(image)

    for i in range(img_h):
        for j in range(img_w):

            region = padded_img[
                i:i + se_h,
                j:j + se_w
            ]

            if np.any(region[se == 1] == 1):
                output[i, j] = 1

    return output


def erosion(image, se):
    """Erosion menggunakan structuring element 3x3."""
    img_h, img_w = image.shape
    se_h, se_w = se.shape

    pad_h = se_h // 2
    pad_w = se_w // 2

    padded_img = np.pad(
        image,
        ((pad_h, pad_h), (pad_w, pad_w)),
        mode='constant',
        constant_values=0
    )

    output = np.zeros_like(image)

    for i in range(img_h):
        for j in range(img_w):

            region = padded_img[
                i:i + se_h,
                j:j + se_w
            ]

            if np.all(region[se == 1] == 1):
                output[i, j] = 1

    return output


def opening(image, se):
    """Opening = Erosion kemudian Dilation."""
    return dilation(erosion(image, se), se)


def closing(image, se):
    """Closing = Dilation kemudian Erosion."""
    return erosion(dilation(image, se), se)


def hit_or_miss_single_se(image, se):
    """
    Hit-or-Miss Transform.

    Kernel:
    1  = Foreground
    0  = Background
    -1 = Don't Care
    """

    img_h, img_w = image.shape
    se_h, se_w = se.shape

    pad_h = se_h // 2
    pad_w = se_w // 2

    padded_img = np.pad(
        image,
        ((pad_h, pad_h), (pad_w, pad_w)),
        mode='constant',
        constant_values=0
    )

    output = np.zeros_like(image)

    for i in range(img_h):
        for j in range(img_w):

            region = padded_img[
                i:i + se_h,
                j:j + se_w
            ]

            mask = (se != -1)

            if np.all(region[mask] == se[mask]):
                output[i, j] = 1

    return output

def setup_image_grid(ax, img, title):
    """Menampilkan citra dengan border setiap piksel."""

    ax.imshow(
        img,
        cmap='gray',
        vmin=0,
        vmax=1,
        interpolation='nearest'
    )

    ax.set_title(
        title,
        fontsize=12,
        fontweight='bold',
        pad=12
    )

    h, w = img.shape

    ax.set_xticks(
        np.arange(-0.5, w, 1),
        minor=True
    )

    ax.set_yticks(
        np.arange(-0.5, h, 1),
        minor=True
    )

    ax.grid(
        which='minor',
        color='gray',
        linestyle='-',
        linewidth=0.9
    )

    ax.tick_params(
        which='minor',
        length=0
    )

    # Nomor koordinat
    ax.set_xticks(range(0, w, 2))
    ax.set_yticks(range(0, h, 2))

    # Border luar
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.2)


def setup_kernel_grid(ax, k, title):
    """Menampilkan kernel 3x3 dengan border kotak."""

    ax.imshow(
        k,
        cmap='gray',
        vmin=0,
        vmax=1,
        interpolation='nearest'
    )

    ax.set_title(
        title,
        fontsize=11,
        fontweight='bold',
        pad=12
    )

    h, w = k.shape

    for i in range(h):
        for j in range(w):

            value = k[i, j]

            ax.text(
                j,
                i,
                str(value),
                ha='center',
                va='center',
                color='black' if value == 1 else 'white',
                fontsize=15,
                fontweight='bold'
            )

    ax.set_xticks(
        np.arange(-0.5, w, 1),
        minor=True
    )

    ax.set_yticks(
        np.arange(-0.5, h, 1),
        minor=True
    )

    ax.grid(
        which='minor',
        color='gray',
        linestyle='-',
        linewidth=1.2
    )

    ax.tick_params(
        which='minor',
        length=0
    )

    ax.set_xticks(range(w))
    ax.set_yticks(range(h))

    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)

    # Border luar kernel
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.2)

sample_image = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0, 1, 0, 0],
    [0, 0, 1, 1, 1, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
], dtype=int)

se_3x3 = np.array([
    [0, 1, 0],
    [1, 1, 1],
    [0, 1, 0]
], dtype=int)

se_hom_single = np.array([
    [0, 1, 1],
    [0, 1, 1],
    [0, 0, 0]
], dtype=int)

def menu_dilasi_erosi():

    print("\n--- Dilation dan Erosion ---")

    res_dil = dilation(sample_image, se_3x3)
    res_ero = erosion(sample_image, se_3x3)

    fig, axes = plt.subplots(
        2, 2,
        figsize=(11, 9)
    )

    fig.suptitle(
        "OPERASI DILATION DAN EROSION",
        fontsize=15,
        fontweight='bold'
    )

    plt.subplots_adjust(
        hspace=0.45,
        wspace=0.35,
        top=0.88
    )

    setup_image_grid(
        axes[0, 0],
        sample_image,
        "Citra Asli"
    )

    setup_image_grid(
        axes[0, 1],
        res_dil,
        "Hasil Dilation"
    )

    setup_image_grid(
        axes[1, 0],
        res_ero,
        "Hasil Erosion"
    )

    setup_kernel_grid(
        axes[1, 1],
        se_3x3,
        "Kernel Plus (+) 3x3"
    )

    filename = "output_dilasi_erosi.png"

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches='tight'
    )

    plt.show()
    plt.close(fig)

    print("Hasil disimpan:", filename)

def menu_opening_closing():

    print("\n--- Opening dan Closing ---")

    res_open = opening(sample_image, se_3x3)
    res_close = closing(sample_image, se_3x3)

    fig, axes = plt.subplots(
        2, 2,
        figsize=(11, 9)
    )

    fig.suptitle(
        "OPERASI OPENING DAN CLOSING",
        fontsize=15,
        fontweight='bold'
    )

    plt.subplots_adjust(
        hspace=0.45,
        wspace=0.35,
        top=0.88
    )

    setup_image_grid(
        axes[0, 0],
        sample_image,
        "Citra Asli"
    )

    setup_image_grid(
        axes[0, 1],
        res_open,
        "Hasil Opening"
    )

    setup_image_grid(
        axes[1, 0],
        res_close,
        "Hasil Closing"
    )

    setup_kernel_grid(
        axes[1, 1],
        se_3x3,
        "Kernel Plus (+) 3x3"
    )

    filename = "output_opening_closing.png"

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches='tight'
    )

    plt.show()
    plt.close(fig)

    print("Hasil disimpan:", filename)

def menu_hit_or_miss():

    print("\n--- Hit-or-Miss Transform ---")

    res_hom = hit_or_miss_single_se(
        sample_image,
        se_hom_single
    )

    fig, axes = plt.subplots(
        1, 3,
        figsize=(15, 5)
    )

    fig.suptitle(
        "OPERASI HIT-OR-MISS TRANSFORM",
        fontsize=15,
        fontweight='bold'
    )

    plt.subplots_adjust(
        wspace=0.35,
        top=0.80
    )

    setup_image_grid(
        axes[0],
        sample_image,
        "Citra Asli"
    )

    setup_image_grid(
        axes[1],
        res_hom,
        "Hasil Hit-or-Miss"
    )

    setup_kernel_grid(
        axes[2],
        se_hom_single,
        "Kernel Hit-or-Miss 3x3"
    )

    filename = "output_hit_or_miss.png"

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches='tight'
    )

    plt.show()
    plt.close(fig)

    print("Hasil disimpan:", filename)

def main():

    while True:

        print("\n==========================================")
        print("      PROGRAM OPERASI MORFOLOGI")
        print("==========================================")
        print("1. Dilation dan Erosion")
        print("2. Opening dan Closing")
        print("3. Hit-or-Miss Transform")
        print("4. Keluar")
        print("==========================================")

        pilihan = input(
            "Pilih menu (1-4): "
        ).strip()

        if pilihan == '1':

            menu_dilasi_erosi()

        elif pilihan == '2':

            menu_opening_closing()

        elif pilihan == '3':

            menu_hit_or_miss()

        elif pilihan == '4':

            print("\nProgram selesai.")
            break

        else:

            print("\nPilihan tidak valid!")

if __name__ == "__main__":
    main()