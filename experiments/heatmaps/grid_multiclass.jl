# Instantiate environment
using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using DrWatson

using SmoothedDifferentiation
using Flux, Metalhead
using Distributions: Normal
using VisionHeatmaps
using ImageNetDataset
using Colors
using ColorTypes: RGB, N0f8
using HTTP, FileIO, ImageIO, ImageMagick

include(joinpath(@__DIR__, "..", "theme.jl"))
set_theme!(theme_smoothdiff())

## Hotfix for Metalhead compatibility
import Flux: loadmodel!
loadmodel!(dst::MaxPool{N, M}, src::Tuple{}; kw...) where {N, M} = dst

## Heatmapping
pipeline = NormReduction() |> PercentileClip() |> ExtremaColormap() |> FlipImage()
figdir = joinpath(@__DIR__, "figures", "multiclass")
!isdir(figdir) && mkdir(figdir)

to_rgb_f32(p::RGB{N0f8}) = RGB{Float32}(convert(Float32, p.r), convert(Float32, p.g), convert(Float32, p.b))

# Load input
url = HTTP.URI("https://raw.githubusercontent.com/Julia-XAI/ExplainableAI.jl/gh-pages/assets/heatmaps/castle.jpg")
img = to_rgb_f32.(load(url))
save(joinpath(figdir, "multiclass_input.png"), load(url))

# Preprocess input
output_size = (224, 224)
mean = (0.485f0, 0.456f0, 0.406f0)
std = (0.229f0, 0.224f0, 0.225f0)
tfm = CenterCropNormalize(; output_size, mean, std)
input = reshape(transform(tfm, img), 224, 224, 3, :)

# Load model
model = VGG(19; pretrain = true).layers

# Load XAI method
n = 10
distr = Normal(0.0f0, 0.5f0)
analyzer = SmoothDiff(model, input, n, distr)

function heatmap_from_index(i)
    expl = analyze(input, analyzer, i)
    return VisionHeatmaps.heatmap(expl, pipeline) |> only
end

## Save individual heatmaps
labels = (
    (484, "castle"),
    (920, "street sign"),
    (437, "station wagon"),
)
for (i, name) in labels
    h = heatmap_from_index(i)
    display(h)
    save(joinpath(figdir, "multiclass_$i.png"), h)
end

## Create plot
function add_img!(fig, img)
    ax = Axis(fig; aspect = DataAspect(), alignmode = Outside())
    image!(ax, rotr90(img); interpolate = false)
    hidedecorations!(ax)
    return ax
end

function add_box!(loc, name, color)
    Box(
        loc;
        color = alphacolor(color, 0.6),
        strokecolor = :white,
        cornerradius = 0,
        strokewidth = 0mm,
    )
    Label(
        loc,
        name;
        fontsize = 2.5mm,
        valign = :center,
        halign = :center,
        rotation = pi / 2,
        padding = (1.5, 1.5, 0, 0),
        tellheight = false,
    )
    return nothing
end

fig = Figure(; size = (8cm, 2.05cm))
g = fig[1, 1] = GridLayout()
add_img!(g[1, 1], load(url))

for (col, (i, name)) in enumerate(labels)
    h = heatmap_from_index(i)
    add_img!(g[1, col + 1], h)
    Label(
        g[1, col + 1, Top()],
        "\"$name\"";
        fontsize = 2.5mm,
        valign = :bottom,
        halign = :center,
    )
end

add_box!(g[1, 1, Left()], "Input", Gray(0.7))
add_box!(g[1, 2, Left()], "SmoothDiff", color_smoothdiff)

colgap!(g, 2)
colgap!(g, 1, 7)
rowgap!(g, 2)

save(joinpath(figdir, "grid_multiclass.png"), fig; px_per_unit = 600 / inch)
display(fig)
