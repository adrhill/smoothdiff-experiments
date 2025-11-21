using Pkg
Pkg.activate(@__DIR__)

using CairoMakie
using ColorSchemes: jet1, batlow, resample, get
using Colors: Lab, colordiff, DE_2000

include(joinpath(@__DIR__, "..", "theme.jl"))

const figdir = joinpath(@__DIR__, "figures")
!isdir(figdir) && mkdir(figdir)

function perceptive_metrics(cs)
    n = length(cs)

    # Compute luminance
    xs = range(0, 1; length = n)
    lum = [c.l for c in convert.(Lab, cs)]

    # Compute Velocity
    dxs = cumsum(diff(xs))
    vel = [colordiff(cs[i], cs[i + 1]; metric = DE_2000()) * (n - 1) for i in 1:(n - 1)]

    return (xs = xs, dxs = dxs, lum = lum, vel = vel, colors = cs.colors)
end

m_jet = perceptive_metrics(jet1)
m_bat = perceptive_metrics(batlow)

function legend_element(cs)
    n = 21
    linepoints = Point2f[(x, 0.5) for x in range(0, 1; length = n)]
    linecolor = resample(cs, n).colors
    elem = [LineElement(; linepoints, linecolor, linewidth = 3)]
    return elem
end

with_theme(theme_smoothdiff()) do
    f = Figure(; size = (LINE_WIDTH_NEURIPS, 4.5cm), figure_padding = 1mm)

    ## Luminance
    ax_l = Axis(f[2, 1]; ylabel = "Luminance\n(CIELab)", xlabel = "Relative position in color map")
    scatterlines!(ax_l, m_bat.xs, m_bat.lum; color = m_bat.colors)
    scatterlines!(ax_l, m_jet.xs, m_jet.lum; color = m_jet.colors)

    ## Velocity
    ax_v = Axis(
        f[2, 2]; ylabel = "Perceptual Velocity\n(CIEDE2000)", xlabel = "Relative position in color map",
        yticks = 0:200:600
    )
    scatterlines!(ax_v, m_bat.dxs, m_bat.vel; color = m_bat.colors)
    scatterlines!(ax_v, m_jet.dxs, m_jet.vel; color = m_jet.colors)

    ## Legend
    Legend(
        f[1, :],
        legend_element.([batlow, jet1]),
        ["Batlow", "Jet"];
        orientation = :horizontal,
        patchsize = (60, 10),
        labelsize = 11,
        rowgap = 10,
    )
    rowgap!(f.layout, 1, 1mm)

    save(joinpath(figdir, "jet_vs_batlow.svg"), f)
    f
end
