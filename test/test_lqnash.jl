@testset "Testing lqnash.jl" begin

    # set up for when agents don't interact with each other
    a = [.95 0.
     0 .95]
    b1 = [.95; 0.]
    b2 = [0.; .95]
    r1 = [-.25 0.
        0. 0.]
    r2 = [0. 0.
        0. -.25]
    q1 = [-.15]
    q2 = [-.15]

    f1, f2, p1, p2 = nnash(a, b1, b2, r1, r2, q1, q2, 0, 0, 0, 0, 0, 0,
                       tol=1e-8, max_iter=10000)

    alq = .95
    blq = .95
    rlq = -.25
    qlq = -.15

    lq_obj = QuantEcon.LQ(qlq, rlq, alq, blq, bet=1)

    p_s, f, d = stationary_values(lq_obj)

    @testset "Checking the policies" begin

        @test isapprox(sum(f1), sum(f2))
        @test isapprox(sum(f1), sum(f))
    end

    @testset "Checking the Value Function" begin

        @test isapprox(p1[1, 1], p2[2, 2])
        @test isapprox(p1[1, 1], p_s[1])
    end

    @testset "Judd test case" begin
        # Define Parameters
        delta = 0.02
        d = [-1 0.5
           0.5 -1]
        B = [25; 25]
        c1 = [1; -2; 1]
        c2 = [1; -2; 1]
        e1 = [10; 10; 3]
        e2 = [10; 10; 3]
        delta_1 = 1 - delta

        ## Define matrices
        a = [delta_1 0 -delta_1*B[1]
           0 delta_1 -delta_1*B[2]
           0 0 1]

        b1 = delta_1 * [1 -d[1, 1]
                      0 -d[2, 1]
                      0 0]
        b2 = delta_1 * [0 -d[1, 2]
                      1 -d[2, 2]
                      0 0]

        r1 = -[0.5*c1[3] 0 0.5*c1[2]
             0 0 0
             0.5*c1[2] 0 c1[1]]
        r2 = -[0 0 0
             0 0.5*c2[3] 0.5*c2[2]
             0 0.5*c2[2] c2[1]]

        q1 = [-0.5*e1[3] 0
            0 d[1, 1]]
        q2 = [-0.5*e2[3] 0
            0 d[2, 2]]

        s1 = zeros(2, 2)
        s2 = copy(s1)

        w1 = [0 0
            0 0
           -0.5*e1[2] B[1]/2.]
        w2 = [0 0
            0 0
           -0.5*e2[2] B[2]/2.]

        m1 = [0 0
            0 d[1, 2]/2.]
        m2 = copy(m1)

        # build model and solve it
        f1, f2, p1, p2 = nnash(a, b1, b2, r1, r2, q1, q2, s1, s2, w1, w2, m1, m2)

        aaa = a - b1*f1 - b2*f2
        aa = aaa[1:2, 1:2]
        tf = I - aa
        tfi = inv(tf)
        xbar = tfi*aaa[1:2, 3]

        # Define answers from matlab. TODO: this is ghetto
        f1_ml = [0.243666582208565   0.027236062661951 -6.827882928738190
                 0.392370733875639   0.139696450885998 -37.734107291009138]

        f2_ml = [0.027236062661951   0.243666582208565  -6.827882928738186
                 0.139696450885998   0.392370733875639 -37.734107291009131]

        xbar_ml = [1.246871007582702, 1.246871007582685]

        @test isapprox(f1, f1_ml)
        @test isapprox(f2, f2_ml)
        @test isapprox(xbar, xbar_ml)

    end
end
