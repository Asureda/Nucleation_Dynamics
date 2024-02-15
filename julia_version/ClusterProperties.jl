module ClusterPhysicsModule

export ClusterPhysics, temperature, activation_energy, diffusivity_factor, sigma, molar_mass,
       jump_distance, mass_density, melting_point, heat_fusion, supersaturation_ratio, method,
       diffusivity, unbiased_jump_rate, molar_volume, molecular_volume, entropy_fusion,
       bulk_free_energy, surface_free_energy, critical_energy_barrier, critical_radius,
       critical_number_of_molecules, total_free_energy,rectified_total_free_energy, rate_equation, number_density_equilibrium,
       stationary_rate, dr_dt

# Constantes
const AVOGADRO = 6.02214076*1e23  # mol^-1
const kB = 1.380649*1e-23  # J/K, constante de Boltzmann

struct ClusterPhysics
    temperature::Float64
    activation_energy::Float64
    diffusivity_factor::Float64
    jump_distance::Float64
    molar_mass::Float64
    mass_density::Float64
    melting_point::Float64
    heat_fusion::Float64
    sigma::Float64
    supersaturation_ratio::Float64
    method::String
end

function ClusterPhysics(params::Dict{String,Any})
    return ClusterPhysics(
        params["temperature"],
        params["activation_energy"],
        params["diffusivity_factor"],
        params["jump_distance"],
        params["molar_mass"],
        params["mass_density"],
        params["melting_point"],
        params["heat_fusion"],
        params["sigma"],
        params["supersaturation_ratio"],
        params["method"]
    )
end

function diffusivity(cp::ClusterPhysics)
    return cp.diffusivity_factor * exp(-cp.activation_energy / (cp.temperature))
end

function unbiased_jump_rate(cp::ClusterPhysics)
    return 6 * diffusivity(cp) / (cp.jump_distance^2)
end

function molar_volume(cp::ClusterPhysics)
    return cp.molar_mass / cp.mass_density
end

function molecular_volume(cp::ClusterPhysics)
    return molar_volume(cp) / AVOGADRO
end

function entropy_fusion(cp::ClusterPhysics)
    return cp.heat_fusion / cp.melting_point
end

function bulk_free_energy(cp::ClusterPhysics)
    if cp.method == "melting"
        return entropy_fusion(cp) * (cp.temperature - cp.melting_point) / AVOGADRO
    elseif cp.method == "saturation"
        S = cp.supersaturation_ratio
        return -(kB * cp.temperature * log(S))
    else
        error("Invalid method. Choose 'melting' or 'saturation'.")
    end
end

function surface_free_energy(cp::ClusterPhysics)
    a = (36π * molecular_volume(cp)^2)^(1/3)
    return a * cp.sigma
end

function critical_energy_barrier(cp::ClusterPhysics)
    return (16*π / 3) * cp.sigma^3 * molecular_volume(cp)^2 / (bulk_free_energy(cp)^2)
end

function critical_radius(cp::ClusterPhysics)
    return -(2 * cp.sigma / (bulk_free_energy(cp) / molecular_volume(cp)))
end

function critical_number_of_molecules(cp::ClusterPhysics)
    return (4/3)*π*critical_radius(cp)^3/molecular_volume(cp)
end

function total_free_energy(cp::ClusterPhysics, number_of_molecules::Int)
    if number_of_molecules < 1
        return 0.0
    else
        return bulk_free_energy(cp) * number_of_molecules +
               surface_free_energy(cp) * number_of_molecules^(2/3)
    end
end

function rectified_total_free_energy(cp::ClusterPhysics, number_of_molecules::Int)
    if number_of_molecules < 1
        return 0.0
    else
        delta_g1 = bulk_free_energy(cp) + surface_free_energy(cp)
        delta_g = bulk_free_energy(cp) * number_of_molecules + surface_free_energy(cp) * number_of_molecules^(2/3)
        return delta_g - delta_g1
    end 
end

function rate_equation(cp::ClusterPhysics, number_of_molecules::Int, attachment=true)
    delta_energy = total_free_energy(cp, number_of_molecules + 1) - total_free_energy(cp, number_of_molecules)
    if attachment
        return 4 * number_of_molecules^(2/3) * unbiased_jump_rate(cp) * exp(-delta_energy / (2 * kB * cp.temperature))
    else
        return 4 * number_of_molecules^(2/3) * unbiased_jump_rate(cp) * exp(delta_energy / (2 * kB * cp.temperature))
    end
end

function number_density_equilibrium(cp::ClusterPhysics, number_of_molecules::Int)
    B1 = AVOGADRO * exp(total_free_energy(cp, 1) / (kB * cp.temperature))
    return AVOGADRO * exp(-total_free_energy(cp, number_of_molecules) / (kB * cp.temperature))
end

end  # Fin del módulo
