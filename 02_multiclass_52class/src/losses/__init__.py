from .multiclass_loss import (
    MulticlassDiceLoss,
    CombinedMulticlassLoss,
    compute_class_weights,
    build_multiclass_loss,
)
from .multiclass_boundary_loss import (
    MulticlassBoundaryLoss,
    BoundaryEnhancedMulticlassLoss,
    build_boundary_multiclass_loss,
    extract_multiclass_boundary,
    build_boundary_weight_map,
)
