using Distributions, Random, Plots
using .MySVM

add_noise = true
add_model_plot = true
add_RMSE_plot = true
max_sv_num = 12
models_sv_num = [3, 6, 7]
models_γ = [4e+3, 3e+4, 1e+5]
models_σ = [0.6, 0.8, 0.7]
RMSE_σ = 0.7
RMSE_γ = 1e+5

function main()
    # Regressor and regressand
    X = LinRange(0, 5.0, 200)
    𝑿 = [[x] for x in X]
    𝒚 = [sinc.(x) for x in X]
    if add_noise
        Random.seed!(666)
        𝒚 += rand(Normal(0, 0.1), length(𝒚))
    end
    𝒚 = transpose(𝒚)

    # Trained SVR model
    function get_SVR_model(𝒦, 𝜷, b, dict_indices)
        return (x) -> b + sum([𝜷[i] * 𝒦(𝑿[dict_indices[i]], [x])
                                        for i=1:length(dict_indices)])
    end

    # Plotting model
    if add_model_plot
        for (sv_num, σ, γ) in zip(models_sv_num, models_σ, models_γ)
            dict_indices, 𝜷, b = MySVM.GSLS_SVM(kernel_RBF(σ), 𝑿, 𝒚, γ, sv_num)
            x = 0:0.01:5.0
            y1 = sinc.(x)
            f = get_SVR_model(kernel_RBF(σ), 𝜷, b, dict_indices)
            y2 = f.(x)
            plot(x, y1, label="theoretical", color="gray", dpi=300)
            plot!(x, y2, label="empirical", color="black")
            scatter!(X, transpose(𝒚),
                                    markersize=2,
                                    markerstrokewidth=0.5,
                                    label="samples",
                                    color="pink")
            scatter!(X[dict_indices], transpose(𝒚)[dict_indices],
                                    markersize=3,
                                    label="support vectors",
                                    color="red")
            savefig("model$sv_num")
        end
    end

    # Plotting dependency of RMSE from number of support vectors
    if add_RMSE_plot
        sv_nums = 1:max_sv_num
        RMSE_vals = []
        for n in sv_nums
            x = 0:0.01:5.0
            dict_indices, 𝜷, b = MySVM.GSLS_SVM(kernel_RBF(RMSE_σ), 𝑿, 𝒚, RMSE_γ, n)
            f = get_SVR_model(MySVM.kernel_RBF(RMSE_σ), 𝜷, b, dict_indices)
            y = f.(transpose(X))
            push!(RMSE_vals, MySVM.RMSE(sinc.(x), f.(x)))
        end
        plot(sv_nums, RMSE_vals, dpi=300,
                                label="RMSE",
                                xticks=1:max_sv_num,
                                xlabel="sv_num",
                                ylabel="RMSE")
        savefig("RMSE.png")
    end
end

main()
