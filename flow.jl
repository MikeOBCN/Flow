using ForwardDiff, OrdinaryDiffEq

function Flow(h)

    function hv(t, x, p, pars)
        n = size(x, 1)
        z = [x ; p]
        foo = z -> h(t, z[1:n], z[n+1:2*n], pars)
        dhdx, dhdp = ForwardDiff.gradient(foo, z)[1:n], ForwardDiff.gradient(foo, z)[n+1:2*n]
        return [ dhdp ; -dhdx ]
    end

    function rhs!(dz, z, pars, t)
        n = size(z, 1)÷2
        x, p = z[1:n], z[n+1:2*n]
        dz[:] = hv(t, x, p, pars)
    end
    
    function f(t0, x0, p0, tf, pars)
        n = size(x0, 1)
        z0 = [ x0 ; p0 ]
        ode = ODEProblem(rhs!, z0, (t0, tf), pars)
        z = solve(ode, Tsit5(), abstol=1e-12, reltol=1e-12, maxiters=1e5) # todo: options with defaults (see args...)
        return z[end][1:n], z[end][n+1:2*n]
    end
    
    return f
    
end
