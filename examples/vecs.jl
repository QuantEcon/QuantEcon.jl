#=
Illustrates vectors in the plane.

@author : Spencer Lyon <spencer.lyon@nyu.edu>

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

# Plot the first figure --- three vectors in the plane

plane_fig()
plt.show()
