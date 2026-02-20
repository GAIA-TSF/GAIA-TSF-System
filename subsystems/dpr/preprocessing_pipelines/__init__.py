class PreprocessingPipelines:
    """The Preprocessing Pipelines are designed as a sequence of
    independent modules tailored for data refinement. Key modules
    include atmospheric correction units, cloud detection services
    (utilizing APIs such as Sen2Cor or Sen2Like), and geometric
    processing tools for georeferencing, orthorectification, and
    reprojection. These pipelines ensure that spatial data is free
    from artifacts and aligned to a common coordinate reference
    system.
    """

    def __init__(self):
        pass
