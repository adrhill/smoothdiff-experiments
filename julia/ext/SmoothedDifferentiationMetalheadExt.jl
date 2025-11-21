module SmoothedDifferentiationMetalheadExt
import SmoothedDifferentiation: prepare
using Metalhead: VGG

prepare(m::VGG, x) = prepare(m.layers, x)
end
