using DifferentialEquations
include("ClusterProperties.jl")  # Asegúrate de que la ruta sea correcta
using .ClusterPhysicsModule

struct JuliaClusterDynamics
    physics_object::ClusterPhysics
    time_steps::Int
    dt::Float64
    u::Int
    i_max::Int
    boundary_type::String
    cluster_array::Vector{Float64}
    number_molecules_array::Vector{Int}
    total_free_energy_array::Vector{Float64}
    forward_rate_array::Vector{Float64}
    backward_rate_array::Vector{Float64}

    function JuliaClusterDynamics(params, time_steps, dt, u, i_max, boundary_type="closed")
        physics_object = ClusterPhysics(params)
        cluster_array = zeros(Float64, i_max)
        number_molecules_array = 1:i_max  # No es necesario convertir a Vector aquí; puede ser útil mantenerlo como rango
        
        # Inicializar arrays directamente para evitar comprensión de listas
        total_free_energy_array = zeros(Float64, i_max)
        forward_rate_array = zeros(Float64, i_max)
        backward_rate_array = zeros(Float64, i_max)
        
        for i in number_molecules_array
            total_free_energy_array[i] = total_free_energy(physics_object, i)
            forward_rate_array[i] = rate_equation(physics_object, i, true)
            backward_rate_array[i] = rate_equation(physics_object, i, false)
        end
        
        for i in 1:u
            cluster_array[i] = number_density_equilibrium(physics_object, i)
        end
    
        new(physics_object, time_steps, dt, u, i_max, boundary_type, cluster_array, collect(number_molecules_array), total_free_energy_array, forward_rate_array, backward_rate_array)
    end
end

function dy_dt!(df, y, p, tg)
    forward_rate_array, backward_rate_array = p
    fill!(df, 0.0)  # Continuar usando fill! para inicializar df con ceros
    len_y = length(y)

    for i in 2:len_y-1
        df[i] = forward_rate_array[i-1] * y[i-1] - 
                (forward_rate_array[i] + backward_rate_array[i]) * y[i] +
                backward_rate_array[i+1] * y[i+1]
    end

    df[end] = -backward_rate_array[end] * y[end] + forward_rate_array[end-1] * y[end-1]

    # Implementación de la condición de frontera (si se requiere).
    #if boundary_type == "open"
    #    df[end] -= forward_rate_array[end] * y[end]
    #end
end

function simulate(dynamics::JuliaClusterDynamics; t_span=(0.0, dynamics.dt * dynamics.time_steps), method=Tsit5(), rtol=1e-3, atol=1e-6)
    prob = ODEProblem(dy_dt!, dynamics.cluster_array, t_span, (dynamics.forward_rate_array, dynamics.backward_rate_array))
    sol = solve(prob, method, reltol=rtol, abstol=atol)
    return sol
end
