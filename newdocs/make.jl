using Documenter, QuantEcon

makedocs(
    modules = [QuantEcon],
    format = Documenter.Formats.HTML,
    sitename = "QuantEcon.jl",
    pages = Any[ # Compat: `Any` for 0.4 compat
        "Home" => "index.md",
        "User Guide" => "man/guide.md",
        "API" => Any[
            "QuantEcon" => "api/QuantEcon.md"
        ]
    ]
)
