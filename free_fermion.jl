module FF

using Statistics;
using LinearAlgebra;
using SkewLinearAlgebra;

export diagonalize_majorana  # Export functions you want to make available
function diagonalize_majorana(M::Matrix{T}) where T
    L = Int(round(size(M,1) / 2))
    Λ,O = schur(M);
    # isapprox(M, O*Λ*Transpose(O))
    Λout = [Λ[2*k-1, 2*k] for k in 1:L]
    return Λout, O
end

export sgn
function sgn(x::Real, tol::Real=1e-10)
    if abs(x) <= tol
        return 0
    else
        return sign(x)
    end
end

export ground_state_correlation
function ground_state_correlation(H::Matrix{T}) where T 
    L = Int(round(size(H,1) / 2));
    Λout, O = diagonalize_majorana(H);
    G = zeros(2L,2L);
    for i in 1:L
        G[2*i-1, 2*i] = -sgn(Λout[i])
        G[2*i, 2*i-1] = sgn(Λout[i])
    end
    G = O*G*Transpose(O)
    gs_energy = -sum(abs.(Λout))
    return G, gs_energy
end

export thermal_state_correlation
function thermal_state_correlation(H::Matrix{T}, β::Float64) where T
    L = Int(round(size(H,1) / 2));
    Λout, O = diagonalize_majorana(H);
    G = zeros(2L,2L);
    for i in 1:L
        G[2*i-1, 2*i] = -tanh(β*0.5*Λout[i])
        G[2*i, 2*i-1] = tanh(β*0.5*Λout[i])
    end
    G = O*G*Transpose(O)
    return G
end

export tfd_correlation
function tfd_correlation(H::Matrix{T}, β::Float64) where T
    N = Int(round(size(H,1) / 2));
    Λout, O = diagonalize_majorana(H);
    ni = -tanh.(β*Λout/2);
    Λ1 = O * kron(diagm(ni),[0 1;-1 0]) * Transpose(O);
    Λ2 = O * kron(diagm(sqrt.(1 .- ni.^2)),[1 0;0 1]) * Transpose(O);
    M_can = [Λ1 Λ2; -Λ2 -Λ1];
    return M_can
    
end


export binary_entropy
function binary_entropy(p::Real, tol = 1e-10)
    # Handle edge cases where p = 0 or 1 (entropy is 0)
    if p < tol || abs(p - 1.0) < tol
        return 0.0
    end
    
    return -p * log(p) - (1-p) * log(1-p)
end

export compute_entropy_majorana
function compute_entropy_majorana(G::Matrix{T}) where T 
    lambdas, ~ = diagonalize_majorana(G);
    # lambdas = [Lambda[i,i+1] for i in 1:2:size(Lambda,1)-1];
    lambdas = (lambdas.+1)./2
    return sum(binary_entropy.(lambdas))
end

export extract_submatrix
function extract_submatrix(M::Matrix, indices::Vector{Int})
    return M[indices, indices]
end

export compute_entanglement_majorana
function compute_entanglement_majorana(G::Matrix{T}, indxA::Vector{Int}) where T
    GA = extract_submatrix(G, indxA)
    SA = compute_entropy_majorana(GA)
    return SA
end

export compute_CMI_majorana
function compute_CMI_majorana(G::Matrix{T}, indxA::Vector{Int}, indxB::Vector{Int}, indxC::Vector{Int}) where T 
    GAC = extract_submatrix(G, [indxA;indxC])
    SAC = compute_entropy_majorana(GAC)
    GBC = extract_submatrix(G, [indxB;indxC])
    SBC = compute_entropy_majorana(GBC)
    GC = extract_submatrix(G, indxC)
    SC = compute_entropy_majorana(GC)
    GABC = extract_submatrix(G, [indxA;indxC;indxB])
    SABC = compute_entropy_majorana(GABC)
    return SAC + SBC - SC- SABC
end

export canonical_purification
function canonical_purification(M::Matrix{T}) where T 
    Λ, R = diagonalize_majorana(M);
    Λ1 = sqrt.(abs.((-Λ.^2).+1)); # abs to get rid of small negative numbers at machine precision
    L = Int(round(size(M,1) / 2));
    M1 = zeros(Float64, 2L, 2L);
    for i in 1:L
        M1[2i-1, 2i-1] = Λ1[i];
        M1[2i,2i] = Λ1[i];
    end
    M1 = R*M1*Transpose(R);

    G = [M M1;
        -M1 -M];
    return G
end

export reduced_dm_CP
function reduced_dm_CP(G::Matrix{T}, site_list::Vector{Int}) where T 
    n = size(site_list, 1);
    L = Int(round(size(G,1) / 4));
    site_list = reshape([site_list; site_list .+ 2L], 2n)
    return extract_submatrix(G, site_list)
end

export compute_CMI_CP_majorana
function compute_CMI_CP_majorana(G::Matrix{T}, indxA::Vector{Int}, indxB::Vector{Int}, indxC::Vector{Int}) where T ## G is the CP correlation matrix
    GAC = reduced_dm_CP(G, [indxA;indxC])
    SAC = compute_entropy_majorana(GAC)
    GBC = reduced_dm_CP(G, [indxB;indxC])
    SBC = compute_entropy_majorana(GBC)
    GC = reduced_dm_CP(G, indxC)
    SC = compute_entropy_majorana(GC)
    GABC = reduced_dm_CP(G, [indxA;indxC;indxB])
    SABC = compute_entropy_majorana(GABC)
    return SAC + SBC - SC- SABC
end


## Ising model

export Ising_Hamiltonian
function Ising_Hamiltonian(N::Int64,g::Float64)
    ## Periodic boundary condition
    H_majorana = zeros(2*N,2*N)
    for i in 1:N
        if(i==1 || i==N)
            H_majorana[2*i-1,2*i] = -g
        else
            H_majorana[2*i-1,2*i] = -g
        end
        if (i<N)
            H_majorana[2*i,2*i+1] = -1
        else
            H_majorana[2N,1] = 1
        end
    end
    H_majorana = H_majorana - H_majorana'
    return H_majorana
end


"""
    generate_random_SO(L::Int)

Generate a random 2L×2L special orthogonal matrix (SO(2L)).
Properties:
- Orthogonal (M'M = MM' = I)
- Determinant = 1
- Size = 2L×2L

Parameters:
- L::Int: Half the size of the matrix (matrix will be 2L×2L)

Returns:
- Matrix{Float64}: A random SO(2L) matrix
"""
function generate_random_SO(L::Int)
    # using LinearAlgebra
    
    # Generate a random matrix
    A = randn(2L, 2L)
    
    # QR decomposition
    Q, R = qr(A)
    
    # Convert Q to a special orthogonal matrix
    Q = Matrix(Q)  # Convert from QR factorization type to matrix
    
    # Ensure determinant is 1
    if det(Q) < 0
        # Multiply first column by -1 to make det = 1
        Q[:, 1] *= -1
    end
    
    return Q
end

"""
    verify_SO(M::Matrix)

Verify that a matrix is in SO(n):
1. Check orthogonality (M'M = MM' = I)
2. Check determinant = 1
3. Check real-valued

Returns:
- Bool: true if all conditions are met
"""
function verify_SO(M::Matrix)
    n = size(M, 1)
    
    # Check if matrix is square
    if size(M, 1) != size(M, 2)
        return false
    end
    
    # Check orthogonality
    is_orthogonal = isapprox(M'M, I(n), rtol=1e-10) && 
                    isapprox(M*M', I(n), rtol=1e-10)
    
    # Check determinant
    has_unit_det = isapprox(det(M), 1.0, rtol=1e-10)
    
    return is_orthogonal && has_unit_det
end

# Test the generation
function test_SO_generation()
    L = 200  # Generate 4×4 matrix
    M = generate_random_SO(L)
    
    println("Generated SO($(2L)) matrix:")
    display(M)
    
    println("\nVerifying properties:")
    println("Is orthogonal: ", isapprox(M'M, I(2L), rtol=1e-10))
    println("Determinant: ", det(M))
    println("All conditions met: ", verify_SO(M))
    
    return M
end

"""
    generate_block_diagonal(L::Int)

Generate a 2L×2L matrix where each 2×2 block along the diagonal
has the form [0 x; -x 0] with x being random in [-1,1].

Parameters:
- L::Int: Half the size of the matrix (matrix will be 2L×2L)

Returns:
- Matrix{Float64}: The block diagonal matrix
"""
function generate_block_diagonal(L::Int)
    # Initialize zero matrix
    M = zeros(2L, 2L)
    
    # Fill in 2×2 blocks along diagonal
    for i in 1:L
        # Calculate indices for current block
        idx1 = 2i-1
        idx2 = 2i
        
        # Generate random x in [-1,1]
        x = 2 * rand() - 1
        
        # Fill in the 2×2 block
        M[idx1, idx2] = x
        M[idx2, idx1] = -x
    end
    
    return M
end

"""
    verify_block_structure(M::Matrix)

Verify that the matrix has the correct block structure:
1. Zeros everywhere except 2×2 blocks on diagonal
2. Each block has form [0 x; -x 0]
3. All x values are in [-1,1]

Returns:
- Bool: true if all conditions are met
"""
function verify_block_structure(M::Matrix)
    L = size(M, 1) ÷ 2
    
    # Check size is even
    if size(M, 1) != size(M, 2) || size(M, 1) % 2 != 0
        return false
    end
    
    # Check each block
    for i in 1:L
        idx1 = 2i-1
        idx2 = 2i
        
        # Check block structure
        if M[idx1, idx1] != 0 || M[idx2, idx2] != 0
            return false
        end
        
        if M[idx1, idx2] != -M[idx2, idx1]
            return false
        end
        
        # Check x is in [-1,1]
        if abs(M[idx1, idx2]) > 1
            return false
        end
    end
    
    # Check that all other elements are zero
    for i in 1:2L
        for j in 1:2L
            block_i = (i-1)÷2 + 1
            block_j = (j-1)÷2 + 1
            if block_i != block_j && M[i,j] != 0
                return false
            end
        end
    end
    
    return true
end

# Test the generation
function test_block_diagonal()
    L = 3
    # Generate 6×6 matrix
    M = generate_block_diagonal(L)
    
    println("Generated block diagonal matrix:")
    display(M)
    
    println("\nVerifying structure: ", verify_block_structure(M))
    
    return M
end

export generate_random_corrMat
function generate_random_corrMat(L::Int)
    R = generate_random_SO(L)
    O = generate_block_diagonal(L)
    return R*O*Transpose(R)
end

end