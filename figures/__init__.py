"""Figure scripts for the paper.

save_png_pdf writes every figure twice: as PNG and as a vector PDF (the format used in
the paper). PDF fonts are embedded as TrueType (Type 42), never Type 3, as required for
the camera-ready version. Image panels (video frames, per-pixel maps, heatmap cells) are
raster inside the PDF; text, axes, lines and markers are vector.
"""
import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


def save_png_pdf(fig, png_path, **kwargs):
    fig.savefig(png_path, **kwargs)
    fig.savefig(png_path[:-4] + ".pdf", **kwargs)
