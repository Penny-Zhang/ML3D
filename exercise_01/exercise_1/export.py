"""Export to disk"""


def export_mesh_to_obj(path, vertices, faces):
    """
    exports mesh as OBJ
    :param path: output path for the OBJ file
    :param vertices: Nx3 vertices
    :param faces: Mx3 faces
    :return: None
    """

    # write vertices starting with "v "
    # write faces starting with "f "

    # ###############
    # TODO: Implement
    with open(path, 'w') as f:
        for v in vertices:
            line = ("v %.1f %.1f %.1f\n" % (v[0], v[1], v[2]))
            f.write(line)
        for face in faces:
            line = ("f %d %d %d\n"% (face[0]+1, face[1]+1, face[2]+1))
            f.write(line)

#     raise NotImplementedError
    # ###############


def export_pointcloud_to_obj(path, pointcloud):
    """
    export pointcloud as OBJ
    :param path: output path for the OBJ file
    :param pointcloud: Nx3 points
    :return: None
    """

    # ###############
    # TODO: Implement
    with open(path, 'w') as f:
        for p in pointcloud:
            line = ("v %.1f %.1f %.1f\n" % (p[0], p[1], p[2]))
            f.write(line)    
    
#     raise NotImplementedError
    # ###############
