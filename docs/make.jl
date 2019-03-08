using Documenter, QuantEcon

makedocs(
    modules = [QuantEcon],
    format = Documenter.HTML(prettyurls = false),
    sitename = "QuantEcon.jl",
    pages = Any[ # Compat: `Any` for 0.4 compat
        "Home" => "index.md",
        # "User Guide" => "man/guide.md",
        "API" => Any[
            "QuantEcon" => "api/QuantEcon.md"
        ],
        "Contributing" => "man/contributing.md"
    ]
)

deploydocs(
    repo = "github.com/QuantEcon/QuantEcon.jl.git",
    branch = "gh-pages",
    target = "build",
    make = nothing,
)
