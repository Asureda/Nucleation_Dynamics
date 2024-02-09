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
    cluster_array::Array{Float64,1}
    number_molecules_array::Array{Int,1}
    total_free_energy_array::Array{Float64,1}
    forward_rate_array::Array{Float64,1}
    backward_rate_array::Array{Float64,1}

    function JuliaClusterDynamics(params, time_steps, dt, u, i_max, boundary_type="closed")
        physics_object = ClusterPhysics(params)
        cluster_array = zeros(Float64, i_max)
        number_molecules_array = collect(1:i_max)  # Usa collect para crear un Vector desde el rango
        # Utiliza comprensión de listas para calcular arrays de energía libre y tasas de reacción
        total_free_energy_array = [total_free_energy(physics_object, i) for i in number_molecules_array]
        forward_rate_array = [rate_equation(physics_object, i, true) for i in number_molecules_array]
        backward_rate_array = [rate_equation(physics_object, i, false) for i in number_molecules_array]
        
        # Cálculo de las densidades de equilibrio para los primeros 'u' números de moléculas y asignación a cluster_array
        for i in 1:u
            cluster_array[i] = number_density_equilibrium(physics_object, i)
        end
    
        new(physics_object, time_steps, dt, u, i_max, boundary_type, cluster_array, number_molecules_array, total_free_energy_array, forward_rate_array, backward_rate_array)
    end
end

function dy_dt(df, y, p, t)
    forward_rate_array, backward_rate_array = p
    # Inicializa df con ceros. Julia automáticamente inferirá el tipo correcto.
    fill!(df, 0.0)
    len_y = length(y)

    # Actualización correcta sin modificar df[1] explícitamente
    for i in 2:len_y-1
        df[i] = forward_rate_array[i-1] * y[i-1] - 
                (forward_rate_array[i] + backward_rate_array[i]) * y[i] +
                backward_rate_array[i+1] * y[i+1]
    end

    # La última actualización se realiza fuera del bucle para el último elemento.
    df[end] = -backward_rate_array[end] * y[end] + forward_rate_array[end-1] * y[end-1]

    # Considera el caso de frontera abierta si es necesario.
    #if boundary_type == "open"
    #    df[end] -= forward_rate_array[end] * y[end]
    #end

    return df
end

function simulate(dynamics::JuliaClusterDynamics; t_span=(0.0, dynamics.dt * dynamics.time_steps), method=Tsit5(), rtol=1e-3, atol=1e-6)
    prob = ODEProblem(dy_dt, dynamics.cluster_array, t_span, (dynamics.forward_rate_array, dynamics.backward_rate_array))
    sol = solve(prob, method, reltol=rtol, abstol=atol)
    return sol;
end
