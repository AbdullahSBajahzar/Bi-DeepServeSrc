function L_cost = cost_loss(cost_used, cost_budget)
% Cost loss (Eq. 16): Squared penalty for exceeding budget
% Input : cost_used   - [N x 1] cost incurred per node
%         cost_budget - [N x 1] budget per node
% Output: L_cost      - scalar

N = length(cost_used);
excess = max(0, cost_used - cost_budget);
L_cost = (1/N) * sum(excess.^2);
end