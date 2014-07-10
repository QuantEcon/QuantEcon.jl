using PyPlot

A = [1.0 2.0
     2.0 1.0]

evals, evecs = eig(A)
evecs =(evecs[:,1], evecs[:,2])

fig, ax = subplots()
for spine in ["left", "bottom"]
    ax[:spines][spine][:set_position]("zero")
end

for spine in ["right", "top"]
    ax[:spines][spine][:set_color]("none")
end
ax[:grid](alpha=0.4)

xmin, xmax = -3, 3
ymin, ymax = -3, 3
ax[:set_xlim](xmin, xmax)
ax[:set_ylim](ymin, ymax)

for v in evecs
    # Plot each eigenvector
    ax[:annotate](" ", xy=v, xytext=[0, 0],
                  arrowprops={"facecolor"=>"blue",
                              "shrink"=>0,
                              "alpha"=>0.6,
                              "width"=>0.5})

    # Plot the image of each eigenvector
    v = A * v
    ax[:annotate](" ", xy=v, xytext=[0, 0],
                  arrowprops={"facecolor"=>"red",
                              "shrink"=>0,
                              "alpha"=>0.6,
                              "width"=>0.5})
end

x = linspace(xmin, xmax, 3)
for v in evecs
    a = v[2] / v[1]
    ax[:plot](x, a .* x, "b-", lw=0.4)
end
