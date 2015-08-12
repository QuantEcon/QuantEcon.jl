# Style Guide

*QuantEcon.jl* maintains a high standard for code quality. To help new contributors write code that meets this standard, this style guide has been developed.

The main goal of the style guide is to help developers write code that is readable and fits with the style of other code across the library. The suggestions in this document are just that: suggestions. If in a particular instance, following every suggestion exactly seems to work against the goal of readable code, submit a pull request that doesn't follow one or more of the guidelines here. Project maintainers will review the request and work with you to find a solution.

The rest of this document is split into various sections. Within each section there is a list of suggestions. When applicable, each list item is followed by one or more examples of "bad" style that violates the suggestion and "good" style that adheres to it.

## Whitespace

- Include spaces after **all** commas. This includes indexing expressions `x[i, j]`, tuples, arguments in function declarations and calls, ect.:
```julia
# bad
y = x[i,j]

# good
y = x[i, j]

# bad
function foo(x::Int,y::Float64)
   ...
end

# good
function foo(x::Int, y::Float64)
   ...
end

# bad
foo(x,y)

# good
foo(x, y)
```
- Always include a new line at the end of every file. Your text editor most likely has a built in mechanism for doing this for you. If you can't find this setting, let us know and we can help you.
- No trailing whitespace at the end of a line (each line should match the regex `.+[^\s]$`). Your editor should also be able to do this for you.
```julia
# bad (note that SPACE below represents a single space)
y = x[i, j]SPACE

# good
y = x[i, j]
```
- Indentation should have 4 spaces. Not 2 spaces, 8, spaces, tabs, ect. You editor should also be able to help here
```julia
# bad - 2 spaces
function foo(x, y)
  x + y
end

# bad - 8 spaces
function foo(x, y)
        x + y
end

# good - 4 spaces
function foo(x, y)
    x + y
end
```
- Do **not** put whitespace around the `=` when declaring function arguments with default values
```julia
# bad
foo(x = 1, y = 3) = 2x + 3y

# good
foo(x=1, y=3) = 2x + 3y
```
- Do **not** put whitespace around `=` when specifying the value of a keyword argument in a function call
```julia
# bad
foo(1.0; y = 2)

# good
foo(1.0; y=2)
```
-  Function arguments that go onto new lines should be indented all the way to the column where the first function argument begins:
```julia
# bad - line to long
function RBLQ(Q::ScalarOrArray, R::ScalarOrArray, A::ScalarOrArray, B::ScalarOrArray, C::ScalarOrArray, bet::Real, theta::Real)
    ...
end

# bad - indentation is just 4 spaces
function RBLQ(Q::ScalarOrArray, R::ScalarOrArray, A::ScalarOrArray,
    B::ScalarOrArray, C::ScalarOrArray, bet::Real, theta::Real)
   ...
end

# good - intendation of `B` lines up with `Q`
function RBLQ(Q::ScalarOrArray, R::ScalarOrArray, A::ScalarOrArray,
              B::ScalarOrArray, C::ScalarOrArray, bet::Real, theta::Real)
   ...
end
```
- Method declarations should be separated by exactly one line of whitespace. The only acceptable exception to this rule is defining multiple one-line functions without whitespace in between (see second pair of examples below)
```julia
# bad - no new line
function foo(x, y)
   ...
end
function bar(z, q)
   ...
end

# bad - more than one new line
function foo(x, y)
   ...
end



function bar(z, q)
   ...
end

# good - one new line
function foo(x, y)
   ...
end

function bar(z, q)
   ...
end
```

```julia
# Acceptable - one line definitions separated with new line
foo(x, y) = ...

bar(z, q) = ...

# better
foo(x, y) = ...
bar(z, q) = ...
```

## Spelling

- Type names should be `UpperCamelCase` (meaning no underscores where the first letter of _every_ word is capitalized)
```julia
# bad - lowerCamelCase
type fooBar
    ...
end

# bad- Upper_Case_With_Underscores
type Foo_Bar
    ...
end

# good - UpperCamelCase
type FooBar
    ...
end
```
- Function names should be all lower case with words separated by underscores (`_`)
```julia
# bad - lowerCamelCase
fooBar(x, y) = ...

# good - lower_case_with_underscores
foo_bar(x, y) = ...
```
