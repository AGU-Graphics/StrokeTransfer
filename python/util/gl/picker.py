def decode_picker(fid_colors):
    fid_img = 256 * 256 * fid_colors[:, :, 0] + 256 * fid_colors[:, :, 1] + fid_colors[:, :, 2]
    return fid_img
