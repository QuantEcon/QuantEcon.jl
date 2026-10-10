using Documenter, QuantEcon
using POMDPs, POMDPTools  # trigger packages of QuantEconPOMDPsExt

const QuantEconPOMDPsExt = Base.get_extension(QuantEcon, :QuantEconPOMDPsExt)

makedocs(
    modules = [QuantEcon, QuantEconPOMDPsExt],
    format = Documenter.HTML(prettyurls = false, size_threshold = nothing),
    sitename = "QuantEcon.jl",
    pages = Any[ # Compat: `Any` for 0.4 compat
        "Home" => "index.md",
        # "User Guide" => "man/guide.md",
        "API" => Any[
            "QuantEcon" => "api/QuantEcon.md",
            "QuantEconPOMDPsExt" => "api/QuantEconPOMDPsExt.md"
        ],
        "Contributing" => "man/contributing.md"
    ]
)

deploydocs(
    repo = "github.com/QuantEcon/QuantEcon.jl.git",
    branch = "gh-pages",
    target = "build",
    make = nothing,
    devbranch = "master",
    versions = ["stable" => "v^", "v#.#", "v#.#.#", "dev" => "dev"],
)
