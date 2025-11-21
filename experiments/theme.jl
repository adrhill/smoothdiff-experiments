using CairoMakie

# Use Wong theme to avoid copyright dispute
const _wong_colors = Makie.wong_colors()

const color_smoothdiff = _wong_colors[1]
const color_smoothgrad = _wong_colors[2]
const color_gradient = _wong_colors[3]
const color_convolution = _wong_colors[6]
const color_beta_smoothing = _wong_colors[4]

# Relative to one CSS pixel
const inch = 96
const pt = 4 / 3
const cm = inch / 2.54
const mm = cm / 10
const LINE_WIDTH_NEURIPS = 5.5inch

firapath(weight) = joinpath(@__DIR__, "fonts", "FiraSans-$(weight).ttf")

const lightfont = firapath("Light")
const lightitalicfont = firapath("LightItalic")
const mediumfont = firapath("Medium")
const mediumitalicfont = firapath("MediumItalic")

function theme_smoothdiff()
    # Modified from AOG theme
    # https://github.com/MakieOrg/AlgebraOfGraphics.jl/blob/76c5245388f6af7da37dc1ea919aedac8ad4224d/src/theme.jl#L32-L130
    # Under MIT license https://github.com/MakieOrg/AlgebraOfGraphics.jl/blob/76c5245388f6af7da37dc1ea919aedac8ad4224d/LICENSE
    # Copyright (c) 2020: Pietro Vertechi.

    return Theme(
        fonts = (
            regular = lightfont,
            italic = lightitalicfont,
            bold = mediumfont,
            bolditalic = mediumitalicfont,
        ),
        fontsize = 11,
        figure_padding = 0,
        marker = :circle,

        colormap = :batlow,
        linecolor = :gray25,
        markercolor = :gray25,
        patchcolor = :gray25,

        palette = (
            color = _wong_colors,
            patchcolor = _wong_colors,
            marker = [:circle, :utriangle, :cross, :rect, :diamond, :dtriangle, :pentagon, :xcross],
            linestyle = [:solid, :dash, :dot, :dashdot, :dashdotdot],
            side = [:left, :right],
        ),

        # setting marker here is a temporary hack
        # it should either respect `marker = :circle` globally
        # or `:circle` and `Circle` should have the same size
        BoxPlot = (mediancolor = :white, marker = :circle),
        Scatter = (marker = :circle,),
        Violin = (mediancolor = :white,),

        Axis = (
            xgridvisible = true, # modified
            ygridvisible = true, # modified
            topspinevisible = false,
            bottomspinevisible = false, # added
            leftspinevisible = false,   # added
            rightspinevisible = false,
            bottomspinecolor = :darkgray,
            leftspinecolor = :darkgray,
            xgridcolor = "#EAEAEA", # added
            ygridcolor = "#EAEAEA", # added
            xtickcolor = :darkgray,
            ytickcolor = :darkgray,
            xminortickcolor = :darkgray,
            yminortickcolor = :darkgray,
            xticklabelfont = lightfont,
            yticklabelfont = lightfont,
            xticklabelsize = 11,
            yticklabelsize = 11,
            xlabelfont = mediumfont,
            ylabelfont = mediumfont,
            xlabelsize = 11,
            ylabelsize = 11,
            titlefont = mediumfont,
            titlesize = 12,
        ),
        Axis3 = (
            protrusions = 55, # to include label on z axis, should be fixed in Makie
            xgridvisible = false,
            ygridvisible = false,
            zgridvisible = false,
            xspinecolor_1 = :darkgray,
            yspinecolor_1 = :darkgray,
            zspinecolor_1 = :darkgray,
            xspinecolor_2 = :transparent,
            yspinecolor_2 = :transparent,
            zspinecolor_2 = :transparent,
            xspinecolor_3 = :transparent,
            yspinecolor_3 = :transparent,
            zspinecolor_3 = :transparent,
            xtickcolor = :darkgray,
            ytickcolor = :darkgray,
            ztickcolor = :darkgray,
            xticklabelfont = lightfont,
            yticklabelfont = lightfont,
            zticklabelfont = lightfont,
            xlabelfont = mediumfont,
            ylabelfont = mediumfont,
            zlabelfont = mediumfont,
            titlefont = mediumfont,
        ),
        Legend = (
            framevisible = false,
            gridshalign = :left,
            padding = (0.0f0, 0.0f0, 0.0f0, 0.0f0),
            labelfont = lightfont,
            titlefont = mediumfont,
        ),
        Colorbar = (
            flip_vertical_label = true,
            spinewidth = 0,
            ticklabelfont = lightfont,
            labelfont = mediumfont,
        ),
    )
end

# set_theme!(theme_smoothdiff())
