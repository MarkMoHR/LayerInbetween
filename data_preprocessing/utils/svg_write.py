from xml.dom import minidom


def write_svg(points_list, height, width, color, svg_save_path, thickness=2.0):
    '''
    Write each stroke as a path.

    Args:
        points_list: list of (N_point, 2), in image size
    '''
    impl_save = minidom.getDOMImplementation()

    doc_save = impl_save.createDocument(None, None, None)

    rootElement_save = doc_save.createElement('svg')
    rootElement_save.setAttribute('xmlns', 'http://www.w3.org/2000/svg')

    rootElement_save.setAttribute('height', str(height) + 'pt')
    rootElement_save.setAttribute('width', str(width) + 'pt')

    view_box = '0 0 ' + str(width) + ' ' + str(height)
    rootElement_save.setAttribute('viewBox', view_box)

    globl_path_i = 0
    for stroke_i, stroke_points in enumerate(points_list):
        # stroke_points: (N_point, 2), in image size
        segment_num = (stroke_points.shape[0] - 1) // 3

        for segment_i in range(segment_num):
            start_idx = segment_i * 3
            start_point = stroke_points[start_idx]
            ctrl_point1 = stroke_points[start_idx + 1]
            ctrl_point2 = stroke_points[start_idx + 2]
            end_point = stroke_points[start_idx + 3]

            command_str = 'M ' + str(start_point[0]) + ', ' + str(start_point[1]) + ' '
            command_str += 'C ' + str(ctrl_point1[0]) + ', ' + str(ctrl_point1[1]) + ' ' \
                           + str(ctrl_point2[0]) + ', ' + str(ctrl_point2[1]) + ' ' \
                           + str(end_point[0]) + ', ' + str(end_point[1]) + ' '

            childElement_save = doc_save.createElement('path')
            childElement_save.setAttribute('id', 'curve_' + str(globl_path_i))
            childElement_save.setAttribute('stroke', color)
            childElement_save.setAttribute('stroke-linejoin', 'round')
            childElement_save.setAttribute('stroke-linecap', 'square')
            childElement_save.setAttribute('fill', 'none')

            childElement_save.setAttribute('d', command_str)
            childElement_save.setAttribute('stroke-width', str(thickness))
            rootElement_save.appendChild(childElement_save)

            globl_path_i += 1

    doc_save.appendChild(rootElement_save)

    f = open(svg_save_path, 'w')
    doc_save.writexml(f, addindent='  ', newl='\n')
    f.close()


def write_svg_chain(points_list, height, width, color, svg_save_path, thickness=2.0):
    '''
    Write each stroke chain as a path.

    Args:
        points_list: list of (N_point, 2), in image size
    '''
    impl_save = minidom.getDOMImplementation()

    doc_save = impl_save.createDocument(None, None, None)

    rootElement_save = doc_save.createElement('svg')
    rootElement_save.setAttribute('xmlns', 'http://www.w3.org/2000/svg')

    rootElement_save.setAttribute('height', str(height) + 'pt')
    rootElement_save.setAttribute('width', str(width) + 'pt')

    view_box = '0 0 ' + str(width) + ' ' + str(height)
    rootElement_save.setAttribute('viewBox', view_box)

    for stroke_i, stroke_points in enumerate(points_list):
        # stroke_points: (N_point, 2), in image size
        segment_num = (stroke_points.shape[0] - 1) // 3

        command_str = 'M ' + str(stroke_points[0][0]) + ', ' + str(stroke_points[0][1]) + ' '

        for segment_i in range(segment_num):
            start_idx = segment_i * 3
            ctrl_point1 = stroke_points[start_idx + 1]
            ctrl_point2 = stroke_points[start_idx + 2]
            end_point = stroke_points[start_idx + 3]

            command_str += 'C ' + str(ctrl_point1[0]) + ', ' + str(ctrl_point1[1]) + ' ' \
                           + str(ctrl_point2[0]) + ', ' + str(ctrl_point2[1]) + ' ' \
                           + str(end_point[0]) + ', ' + str(end_point[1]) + ' '

        childElement_save = doc_save.createElement('path')
        childElement_save.setAttribute('id', 'curve_' + str(stroke_i))
        childElement_save.setAttribute('stroke', color)
        childElement_save.setAttribute('stroke-linejoin', 'round')
        childElement_save.setAttribute('stroke-linecap', 'square')
        childElement_save.setAttribute('fill', 'none')

        childElement_save.setAttribute('d', command_str)
        childElement_save.setAttribute('stroke-width', str(thickness))
        rootElement_save.appendChild(childElement_save)

    doc_save.appendChild(rootElement_save)

    f = open(svg_save_path, 'w')
    doc_save.writexml(f, addindent='  ', newl='\n')
    f.close()
