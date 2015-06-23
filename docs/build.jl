using Docile, Lexicon, QuantEcon

const api_directory = "api"
const modules = [QuantEcon, QuantEcon.Models]

# const nb_src_dir = joinpath(dirname(dirname(@__FILE__)), "solutions")
# const nb_output_dir = joinpath(dirname(dirname(@__FILE__)), "solutions")
# const nb_list = filter(x->endswith(x, "ipynb"), readdir(nb_src_dir))

cd(dirname(@__FILE__)) do
    # Generate and save the contents of docstrings as markdown files.
    index  = Index()
    for mod in modules
        Lexicon.update!(index, save(joinpath(api_directory, "$(mod).md"), mod))
    end
    save(joinpath(api_directory, "index.md"), index; md_subheader = :category)

    # Add a reminder not to edit the generated files.
    open(joinpath(api_directory, "README.md"), "w") do f
        print(f, """
        Files in this directory are generated using the `build.jl` script. Make
        all changes to the originating docstrings/files rather than these ones.
        Documentation should *only* be built directly on the `master` branch.
        Source links would otherwise become unavailable should a branch be
        deleted from the `origin`. This means potential pull request authors
        *should not* run the build script when filing a PR.
        """)
    end

    # for nb in nb_list
    #     nb_path = joinpath(nb_src_dir, nb)
    #     md_path = joinpath(nb_output_dir, replace(nb, ".ipynb", ".md"))
    #     run(`ipython nbconvert --to markdown $nb_path --output=$md_path`)
    # end

    # info("Adding all documentation changes in $(api_directory) to this commit.")
    # success(`git add $(api_directory)`) || exit(1)

end


cd(dirname(dirname(@__FILE__))) do
    yaml = """
    site_name: QuantEcon.jl
    site_description: QuantEcon.jl Quantitative economics in Julia
    site_author: Spencer Lyon
    repo_name: GitHub
    site_favicon: favicon.ico
    docs_dir: 'docs'
    site_dir: 'site'
    repo_url: https://github.com/QuantEcon/QuantEcon.jl/

    pages:
    - Home: 'index.md'
    - Overview: 'api/index.md'
    - API Docs:
      - QuantEcon: 'api/QuantEcon.md'
      - Models: 'api/QuantEcon.Models.md'
    """

    # TODO: add the solutions if I figure out how to get them to render properly

    open("mkdocs.yml", "w") do f
        write(f, yaml)
    end
end
