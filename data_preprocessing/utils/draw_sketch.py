import cv2
import numpy as np
import cairocffi as cairo
from PIL import Image

from .colors import generate_colors2


def draw_sketch_cairo(stroke_data, outpath, is_bezier,
                      part_label=False, parts_used=None, vis_color=None, clip_curve=False,
                      side=64, line_diameter=16,
                      bg_color=(1, 1, 1), fg_color=(0, 0, 0)):
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, side, side)
    ctx = cairo.Context(surface)
    ctx.set_antialias(cairo.ANTIALIAS_BEST)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_line_width(line_diameter)

    # clear background
    ctx.set_source_rgb(*bg_color)
    ctx.paint()

    # draw strokes, this is the most cpu-intensive part
    ctx.set_source_rgb(*fg_color)
    for j, component in enumerate(stroke_data):
        if part_label:
            assert parts_used is not None
            assert vis_color is not None
            ctx.set_source_rgb(*vis_color[parts_used[j]])
        for curve in component:  # (N, 2) for polyline, (N, 4, 2) for bezier
            if clip_curve:
                curve = np.clip(curve, 1.0, side - 1)
            # if len(curve) == 0:
            #     continue
            assert len(curve) > 0
            if not is_bezier:
                ctx.move_to(curve[0][0], curve[0][1])
                for x, y in curve:
                    ctx.line_to(x, y)
            else:
                for si in range(len(curve)):
                    stroke_i = curve[si]
                    p0, p1, p2, p3 = stroke_i
                    x0, y0 = p0
                    x1, y1 = p1
                    x2, y2 = p2
                    x3, y3 = p3
                    ctx.move_to(x0, y0)
                    ctx.curve_to(x1, y1, x2, y2, x3, y3)
            ctx.stroke()
    surface_data = surface.get_data()
    if part_label:
        raster_image = np.copy(np.asarray(surface_data)).reshape(side, side, 4)[:, :, :3]
    else:
        raster_image = np.copy(np.asarray(surface_data))[::4].reshape(side, side)

    if outpath is not None:
        cv2.imwrite(outpath, raster_image)
    return raster_image


def draw_sketch_hybrid_cairo(stroke_data_bezier, stroke_data_line, outpath,
                             part_label=False, parts_used=None, vis_color=None, clip_curve=False,
                             side=64, line_diameter=16,
                             bg_color=(1, 1, 1), fg_color=(0, 0, 0)):
    '''
    Draw polylines and bezier curves together
    '''
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, side, side)
    ctx = cairo.Context(surface)
    ctx.set_antialias(cairo.ANTIALIAS_BEST)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_line_width(line_diameter)

    # clear background
    ctx.set_source_rgb(*bg_color)
    ctx.paint()

    # draw strokes, this is the most cpu-intensive part
    ctx.set_source_rgb(*fg_color)
    for j, component in enumerate(stroke_data_bezier):
        if part_label:
            assert parts_used is not None
            assert vis_color is not None
            ctx.set_source_rgb(*vis_color[parts_used[j]])
        for curve in component:  # (N, 4, 2) for bezier
            if clip_curve:
                curve = np.clip(curve, 1.0, side - 1)
            assert len(curve) > 0
            for si in range(len(curve)):
                stroke_i = curve[si]
                p0, p1, p2, p3 = stroke_i
                x0, y0 = p0
                x1, y1 = p1
                x2, y2 = p2
                x3, y3 = p3
                ctx.move_to(x0, y0)
                ctx.curve_to(x1, y1, x2, y2, x3, y3)
            ctx.stroke()
    for j, component in enumerate(stroke_data_line):
        if part_label:
            assert parts_used is not None
            assert vis_color is not None
            ctx.set_source_rgb(*vis_color[parts_used[j]])
        for curve in component:  # (N, 2) for polyline
            assert len(curve) > 0
            ctx.move_to(curve[0][0], curve[0][1])
            for x, y in curve:
                ctx.line_to(x, y)
            ctx.stroke()
    surface_data = surface.get_data()
    if part_label:
        raster_image = np.copy(np.asarray(surface_data)).reshape(side, side, 4)[:, :, :3]
    else:
        raster_image = np.copy(np.asarray(surface_data))[::4].reshape(side, side)

    if outpath is not None:
        cv2.imwrite(outpath, raster_image)
    return raster_image


def draw_sketch_stroke_cairo(stroke_data, outpath, is_bezier,
                             side=64, line_diameter=16,
                             bg_color=(1, 1, 1), bg_sketch=None,
                             max_seq_number=None, color_shift=0):
    '''
    bg_sketch: (H, W), [0-stroke, 255-BG]
    '''
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, side, side)
    ctx = cairo.Context(surface)
    ctx.set_antialias(cairo.ANTIALIAS_BEST)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_line_width(line_diameter)

    # clear background
    ctx.set_source_rgb(*bg_color)
    ctx.paint()

    if max_seq_number is None:
        max_seq_number = 0
        for j, component in enumerate(stroke_data):
            for curve in component:  # (N, 2) for polyline, (N, 4, 2) for bezier
                max_seq_number += len(curve)
    colors = generate_colors2(max_seq_number)  # list of (3), in [0., 1.]

    seq_i = 0
    for j, component in enumerate(stroke_data):
        for curve in component:  # (N, 2) for polyline, (N, 4, 2) for bezier
            assert len(curve) > 0
            if not is_bezier:
                for si in range(len(curve) - 1):
                    x0, y0 = curve[si]
                    x1, y1 = curve[si + 1]
                    ctx.set_source_rgb(colors[seq_i + color_shift][0], colors[seq_i + color_shift][1], colors[seq_i + color_shift][2])
                    seq_i += 1
                    ctx.move_to(x0, y0)
                    ctx.line_to(x1, y1)
                    ctx.stroke()
            else:
                for si in range(len(curve)):
                    stroke_i = curve[si]
                    p0, p1, p2, p3 = stroke_i
                    x0, y0 = p0
                    x1, y1 = p1
                    x2, y2 = p2
                    x3, y3 = p3
                    ctx.set_source_rgb(colors[seq_i + color_shift][0], colors[seq_i + color_shift][1], colors[seq_i + color_shift][2])
                    seq_i += 1
                    ctx.move_to(x0, y0)
                    ctx.curve_to(x1, y1, x2, y2, x3, y3)
                    ctx.stroke()
    surface_data = surface.get_data()
    raster_image_rgb = np.copy(np.asarray(surface_data)).reshape(side, side, 4)[:, :, :3]

    if bg_sketch is not None:
        # add baclground
        raster_image_rgb_bg = np.copy(bg_sketch).astype(np.float32)  # (H, W), [0-stroke, 255-BG]
        raster_image_rgb_bg = np.tile(np.expand_dims(raster_image_rgb_bg, axis=-1), (1, 1, 3))
        raster_image_rgb_bg = 255 - (255 - raster_image_rgb_bg) * 0.2
        rgb_mask = (raster_image_rgb != 255).any(-1)
        raster_image_rgb_bg[rgb_mask] = raster_image_rgb[rgb_mask]
        raster_image_rgb_out = raster_image_rgb_bg.astype(np.uint8)
    else:
        raster_image_rgb_out = raster_image_rgb

    if outpath is not None:
        raster_image_png = Image.fromarray(raster_image_rgb_out, 'RGB')
        raster_image_png.save(outpath, 'PNG')
    return raster_image_rgb


def draw_sketch_stroke_chain_cairo(stroke_data, outpath, is_bezier,
                                   side=64, line_diameter=16,
                                   bg_color=(1, 1, 1)):
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, side, side)
    ctx = cairo.Context(surface)
    ctx.set_antialias(cairo.ANTIALIAS_BEST)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_line_width(line_diameter)

    # clear background
    ctx.set_source_rgb(*bg_color)
    ctx.paint()

    max_seq_number = len(stroke_data)
    colors = generate_colors2(max_seq_number)  # list of (3), in [0., 1.]
    if max_seq_number == 2:
        colors = [[0., 0., 0.], [0.0, 0.0, 1.0]]

    for j, component in enumerate(stroke_data):
        ctx.set_source_rgb(colors[j][0], colors[j][1], colors[j][2])
        for curve in component:  # (N, 2) for polyline, (N, 4, 2) for bezier
            assert len(curve) > 0
            if not is_bezier:
                for si in range(len(curve) - 1):
                    x0, y0 = curve[si]
                    x1, y1 = curve[si + 1]
                    ctx.move_to(x0, y0)
                    ctx.line_to(x1, y1)
                    ctx.stroke()
            else:
                for si in range(len(curve)):
                    stroke_i = curve[si]
                    p0, p1, p2, p3 = stroke_i
                    x0, y0 = p0
                    x1, y1 = p1
                    x2, y2 = p2
                    x3, y3 = p3
                    ctx.move_to(x0, y0)
                    ctx.curve_to(x1, y1, x2, y2, x3, y3)
                    ctx.stroke()
    surface_data = surface.get_data()
    raster_image = np.copy(np.asarray(surface_data)).reshape(side, side, 4)[:, :, :3]
    if outpath is not None:
        raster_image_png = Image.fromarray(raster_image, 'RGB')
        raster_image_png.save(outpath, 'PNG')
    return raster_image


def draw_highlight_stroke_cairo(stroke_data, highlight_status, outpath, is_bezier,
                                side=64, line_diameter=16,
                                bg_color=(1, 1, 1)):
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, side, side)
    ctx = cairo.Context(surface)
    ctx.set_antialias(cairo.ANTIALIAS_BEST)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_line_width(line_diameter)

    # clear background
    ctx.set_source_rgb(*bg_color)
    ctx.paint()

    for j, component in enumerate(stroke_data):
        highlight_status_c = highlight_status[j]
        for curve_i, curve in enumerate(component):  # (N, 2) for polyline, (N, 4, 2) for bezier
            highlight_status_curve = highlight_status_c[curve_i]  # (N)

            assert len(curve) > 0
            if not is_bezier:
                for si in range(len(curve) - 1):
                    x0, y0 = curve[si]
                    x1, y1 = curve[si + 1]
                    color = (0, 0, 1) if highlight_status_curve[si] else (0, 0, 0)
                    ctx.set_source_rgb(*color)
                    ctx.move_to(x0, y0)
                    ctx.line_to(x1, y1)
                    ctx.stroke()
            else:
                for si in range(len(curve)):
                    stroke_i = curve[si]
                    p0, p1, p2, p3 = stroke_i
                    x0, y0 = p0
                    x1, y1 = p1
                    x2, y2 = p2
                    x3, y3 = p3
                    color = (0, 0, 1) if highlight_status_curve[si] else (0, 0, 0)
                    ctx.set_source_rgb(*color)
                    ctx.move_to(x0, y0)
                    ctx.curve_to(x1, y1, x2, y2, x3, y3)
                    ctx.stroke()
    surface_data = surface.get_data()
    raster_image = np.copy(np.asarray(surface_data)).reshape(side, side, 4)[:, :, :3]
    if outpath is not None:
        raster_image_png = Image.fromarray(raster_image, 'RGB')
        raster_image_png.save(outpath, 'PNG')
    return raster_image
