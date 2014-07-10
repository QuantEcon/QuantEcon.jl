#=
Illustrates vectors in the plane.

@authors: Spencer Lyon, Tom Sargent, John Stachurski
@date: 07/09/2014
=#

using PyPlot


#=
    Create pyplot figure and axis. Move left and bottom spines to
    intersect at origin. Remove right and top spines. Return the axis
=#
function move_spines()
    fix, ax = subplots()
    for spine in ["left", "bottom"]
        ax[:spines][spine][:set_position]("zero")
    end

    for spine in ["right", "top"]
        ax[:spines][spine][:set_color]("none")
    end
    return ax
end


function plane_fig()  # illustrate vectors in a plane
    ax = move_spines()

    ax[:set_xlim](-5, 5)
    ax[:set_ylim](-5, 5)
    ax[:grid]()
    vecs = {[2, 4], [-3, 3], [-4, -3.5]}
    for v in vecs
        ax[:annotate](" ", xy=v, xytext=[0, 0],
                    arrowprops={"facecolor"=>"blue",
                                "shrink"=>0,
                                "alpha"=>0.7,
                                "width"=>0.5})
        ax[:text](1.1 * v[1], 1.1 * v[2], string(v))
    end
end


function scalar_multiply()  # illustrate scalar multiplication
    ax = move_spines()
    ax[:set_xlim](-5, 5)
    ax[:set_ylim](-5, 5)

    x = [2, 2]
    ax[:annotate](" ", xy=x, xytext=[0, 0],
                  arrowprops={"facecolor"=>"blue",
                              "shrink"=>0,
                              "alpha"=>1,
                              "width"=>0.5})

    ax[:text](x[1] + 0.4, x[2] - 0.2, L"$x$", fontsize="16")

    scalars = [-2, 2]

    for s in scalars
        v = s .* x
        ax[:annotate](" ", xy=v, xytext=[0, 0],
                      arrowprops={"facecolor"=>"red",
                                  "shrink"=>0,
                                  "alpha"=>0.5,
                                  "width"=>0.5})

        ax[:text](v[1] + 0.4, v[2] - 0.2, LaTeXString("\$$s x\$"),
                  fontsize="16")
    end
end


function span_3d()  # Illustrates the span of two vectors in R^3.
    fig = figure()
    ax = fig[:gca](projection="3d")

    x_min, x_max = -5, 5
    y_min, y_max = -5, 5

    alpha, bet = 0.2, 0.1

    ax[:set_xlim]((x_min, x_max))
    ax[:set_ylim]((x_min, x_max))
    ax[:set_zlim]((x_min, x_max))

    ax[:set_xticks]([0])
    ax[:set_yticks]([0])
    ax[:set_zticks]([0])
    gs = 3
    z = linspace(x_min, x_max, gs)
    x = zeros(gs)
    y = zeros(gs)
    ax[:plot](x, y, z, "k-", lw=2, alpha=0.5)
    ax[:plot](z, x, y, "k-", lw=2, alpha=0.5)
    ax[:plot](y, z, x, "k-", lw=2, alpha=0.5)

    # Fixed linear function, to generate a plane
    f(x, y) = alpha .* x .+ bet .* y

    # Vector locations, by coordinate
    x_coords = [3, 3]
    y_coords = [4, -4]
    z = f(x_coords, y_coords)

    for i=1:2
        ax[:text](x_coords[i], y_coords[i], z[i],
                  LaTeXString("\$a_{$i} \$"), fontsize=14)
        x = [0, x_coords[i]]
        y = [0, y_coords[i]]
        z = [0, f(x[2], y[2])]
        ax[:plot](x, y, z, "b-", lw=1.5, alpha=0.6)
    end

    # Draw the plane
    grid_size = 20
    xr2 = linspace(x_min, x_max, grid_size)
    yr2 = linspace(y_min, y_max, grid_size)
    z2 = f(xr2', yr2)
    ax[:plot_surface](xr2'', yr2, z2, rstride=1, cstride=1, cmap=ColorMap("jet"),
            linewidth=0, antialiased=true, alpha=0.2)
    ax[:set_zlabel]("z")
end






