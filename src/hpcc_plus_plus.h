/**
 * HPCC++ Rate Control Implementation
 * Based on IETF Draft and Enhanced Control Theory
 * 
 * Author: Xhovani Mali
 * Course: ECE-GY 6383 High-Speed Networks
 * 
 * This can be integrated with htsim or ns-3 simulators
 */

#ifndef HPCC_PLUS_PLUS_H
#define HPCC_PLUS_PLUS_H

#include <cmath>
#include <algorithm>
#include <vector>
#include <iostream>

namespace hpcc {

/**
 * Link telemetry information (equivalent to C-SIG)
 * Carried in-band with packets via INT (In-band Network Telemetry)
 */
struct LinkTelemetry {
    double qLen;        // Queue length in bytes
    double txBytes;     // Transmitted bytes in this interval
    double capacity;    // Link capacity in bps
    double timestamp;   // Timestamp in seconds
    int switchId;       // Switch identifier
    int portId;         // Output port identifier
    
    /**
     * Compute normalized utilization U_j
     * U_j = qLen/(B*T) + txRate/B
     */
    double utilization(double baseRTT) const {
        double qRatio = qLen / (capacity * baseRTT);
        double txRate = txBytes / baseRTT;
        double rateRatio = txRate / capacity;
        return qRatio + rateRatio;
    }
};

/**
 * HPCC++ control parameters
 */
struct HPCCParams {
    double alpha;       // Responsiveness to utilization error (default: 0.1)
    double beta;        // Damping coefficient (default: 0.02)
    double eta;         // Target utilization (default: 0.95)
    double W_AI;        // Additive increase in bytes (default: 1000)
    double T_s;         // Feedback sampling interval in seconds (default: 0.01)
    double minWindow;   // Minimum window size
    double maxWindow;   // Maximum window size
    
    HPCCParams() 
        : alpha(0.1), beta(0.02), eta(0.95), W_AI(1000.0), T_s(0.01),
          minWindow(0), maxWindow(0) {}
    
    /**
     * Check stability constraints
     * Constraint 1: alpha * C * T_s < 1
     * Constraint 2: beta >= alpha / 2 (critical damping)
     */
    bool checkStability(double capacity, std::string& msg) const {
        // Constraint 1
        double constraint1 = alpha * capacity * T_s;
        if (constraint1 >= 1.0) {
            msg = "Stability violated: alpha*C*T_s = " + 
                  std::to_string(constraint1) + " >= 1";
            return false;
        }
        
        // Constraint 2
        if (beta < alpha / 2.0) {
            msg = "Insufficient damping: beta = " + std::to_string(beta) + 
                  " < alpha/2 = " + std::to_string(alpha/2.0);
            return false;
        }
        
        msg = "Parameters stable";
        return true;
    }
};

/**
 * HPCC++ Flow Controller
 * Implements rate-based congestion control using explicit telemetry feedback
 */
class HPCCFlow {
private:
    HPCCParams params_;
    double capacity_;           // Link capacity in bps
    double baseRTT_;           // Base propagation RTT in seconds
    double window_;            // Current congestion window in bytes
    double rate_;              // Current sending rate in bps
    
    // State tracking
    double prevUtilization_;   // Previous utilization for derivative
    double prevTimestamp_;     // Previous update timestamp
    
    // Statistics
    std::vector<double> windowHistory_;
    std::vector<double> rateHistory_;
    std::vector<double> utilizationHistory_;
    std::vector<double> queueHistory_;
    
public:
    /**
     * Constructor
     * @param params Control parameters
     * @param capacity Link capacity in bps
     * @param baseRTT Base RTT in seconds
     */
    HPCCFlow(const HPCCParams& params, double capacity, double baseRTT)
        : params_(params), capacity_(capacity), baseRTT_(baseRTT),
          prevUtilization_(0), prevTimestamp_(0) {
        
        // Initialize window to bandwidth-delay product
        window_ = capacity * baseRTT;
        rate_ = capacity;
        
        // Set min/max window bounds if not specified
        if (params_.minWindow == 0) {
            params_.minWindow = 0.1 * window_;
        }
        if (params_.maxWindow == 0) {
            params_.maxWindow = 2.0 * window_;
        }
        
        windowHistory_.push_back(window_);
        rateHistory_.push_back(rate_);
    }
    
    /**
     * Update rate based on link telemetry feedback
     * This is the core HPCC++ control algorithm
     * 
     * @param telemetry Current link state from INT
     * @param hasPrevTelemetry Whether we have previous telemetry for derivative
     * @param prevTelemetry Previous link state (for derivative calculation)
     */
    void updateRate(const LinkTelemetry& telemetry, 
                   bool hasPrevTelemetry = false,
                   const LinkTelemetry* prevTelemetry = nullptr) {
        
        // Compute current normalized utilization U_j
        double U_j = telemetry.utilization(baseRTT_);
        utilizationHistory_.push_back(U_j);
        queueHistory_.push_back(telemetry.qLen);
        
        // Compute derivative term for damping
        double dU_dt = 0.0;
        if (hasPrevTelemetry && prevTelemetry != nullptr) {
            double U_j_prev = prevTelemetry->utilization(baseRTT_);
            double dt = telemetry.timestamp - prevTelemetry->timestamp;
            if (dt > 0) {
                dU_dt = (U_j - U_j_prev) / dt;
            }
        }
        
        // HPCC++ enhanced control law
        // W_i(t+1) = W_i(t) * [1 - α(U_j - η) - β*dU_j/dt] + W_AI
        
        double utilizationError = U_j - params_.eta;
        double dampingTerm = params_.beta * dU_dt;
        
        // Multiplicative factor
        double multFactor = 1.0 - params_.alpha * utilizationError - dampingTerm;
        
        // Clamp multiplicative factor to prevent extreme changes
        multFactor = std::max(0.5, std::min(1.5, multFactor));
        
        // Update window
        double newWindow = window_ * multFactor + params_.W_AI;
        
        // Clamp window to bounds
        window_ = std::max(params_.minWindow, 
                          std::min(params_.maxWindow, newWindow));
        
        // Compute rate from window: R = W / RTT
        rate_ = window_ / baseRTT_;
        
        // Record history
        windowHistory_.push_back(window_);
        rateHistory_.push_back(rate_);
        
        prevUtilization_ = U_j;
        prevTimestamp_ = telemetry.timestamp;
    }
    
    /**
     * Get current sending rate
     */
    double getRate() const { return rate_; }
    
    /**
     * Get current window size
     */
    double getWindow() const { return window_; }
    
    /**
     * Compute performance metrics
     */
    struct Metrics {
        double avgRate;
        double rateStdDev;
        double avgUtilization;
        double avgQueue;
        double maxQueue;
        double utilizationStdDev;
    };
    
    Metrics getMetrics() const {
        Metrics m;
        
        // Compute averages
        m.avgRate = computeMean(rateHistory_);
        m.rateStdDev = computeStdDev(rateHistory_, m.avgRate);
        m.avgUtilization = computeMean(utilizationHistory_);
        m.utilizationStdDev = computeStdDev(utilizationHistory_, m.avgUtilization);
        m.avgQueue = computeMean(queueHistory_);
        
        // Find max queue
        m.maxQueue = 0;
        for (double q : queueHistory_) {
            if (q > m.maxQueue) m.maxQueue = q;
        }
        
        return m;
    }
    
    /**
     * Print current state (for debugging)
     */
    void printState() const {
        std::cout << "HPCC++ Flow State:" << std::endl;
        std::cout << "  Window: " << window_ / 1e6 << " MB" << std::endl;
        std::cout << "  Rate: " << rate_ / 1e9 << " Gbps" << std::endl;
        if (!utilizationHistory_.empty()) {
            std::cout << "  Utilization: " << utilizationHistory_.back() << std::endl;
        }
        if (!queueHistory_.empty()) {
            std::cout << "  Queue: " << queueHistory_.back() / 1024 << " KB" << std::endl;
        }
    }

private:
    /**
     * Helper: compute mean of vector
     */
    static double computeMean(const std::vector<double>& vec) {
        if (vec.empty()) return 0.0;
        double sum = 0.0;
        for (double v : vec) sum += v;
        return sum / vec.size();
    }
    
    /**
     * Helper: compute standard deviation
     */
    static double computeStdDev(const std::vector<double>& vec, double mean) {
        if (vec.size() <= 1) return 0.0;
        double sum = 0.0;
        for (double v : vec) {
            double diff = v - mean;
            sum += diff * diff;
        }
        return std::sqrt(sum / (vec.size() - 1));
    }
};

/**
 * Parameter tuner using grid search
 */
class ParameterTuner {
public:
    struct TuningResult {
        double alpha;
        double beta;
        double eta;
        double cost;
        HPCCFlow::Metrics metrics;
    };
    
    /**
     * Objective function for optimization
     * Lower is better
     */
    static double objectiveFunction(const HPCCFlow::Metrics& metrics,
                                   double targetUtil = 0.95,
                                   double w_queue = 1.0,
                                   double w_util = 1.0,
                                   double w_stability = 1.0) {
        // Queue cost (normalized)
        double queueCost = w_queue * (metrics.avgQueue / 1e6);
        
        // Utilization error cost
        double utilCost = w_util * std::abs(metrics.avgUtilization - targetUtil);
        
        // Stability cost (rate variance)
        double stabilityCost = w_stability * (metrics.rateStdDev / 1e9);
        
        return queueCost + utilCost + stabilityCost;
    }
    
    /**
     * Grid search over parameter space
     * Returns best parameters found
     */
    static TuningResult gridSearch(
        const std::vector<double>& alphaValues,
        const std::vector<double>& betaValues,
        double eta,
        double capacity,
        double baseRTT) {
        
        TuningResult best;
        best.cost = 1e9;  // Initialize to very high cost
        
        std::cout << "Running grid search over " 
                  << alphaValues.size() * betaValues.size() 
                  << " parameter combinations..." << std::endl;
        
        for (double alpha : alphaValues) {
            for (double beta : betaValues) {
                HPCCParams params;
                params.alpha = alpha;
                params.beta = beta;
                params.eta = eta;
                
                // Check stability
                std::string msg;
                if (!params.checkStability(capacity, msg)) {
                    continue;  // Skip unstable parameters
                }
                
                // Would run simulation here and evaluate
                // For demonstration, we'll just record valid parameters
                // In real implementation, integrate with your simulator
                
                TuningResult result;
                result.alpha = alpha;
                result.beta = beta;
                result.eta = eta;
                
                // Placeholder: actual simulation would compute these
                // result.metrics = runSimulation(params, capacity, baseRTT);
                // result.cost = objectiveFunction(result.metrics, eta);
                
                // if (result.cost < best.cost) {
                //     best = result;
                // }
            }
        }
        
        return best;
    }
};

} // namespace hpcc

#endif // HPCC_PLUS_PLUS_H

/**
 * Example usage / test harness
 */
#ifdef HPCC_TEST_MAIN

#include <iostream>
#include <iomanip>

int main() {
    using namespace hpcc;
    
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "HPCC++ Parameter Tuning Test" << std::endl;
    std::cout << "=============================" << std::endl << std::endl;
    
    // Network configuration
    double capacity = 100e9;    // 100 Gbps
    double baseRTT = 10e-6;     // 10 microseconds
    
    // Test baseline parameters
    HPCCParams baseline;
    baseline.alpha = 0.1;
    baseline.beta = 0.02;
    baseline.eta = 0.95;
    baseline.T_s = 0.01;
    
    std::cout << "Network Configuration:" << std::endl;
    std::cout << "  Capacity: " << capacity / 1e9 << " Gbps" << std::endl;
    std::cout << "  Base RTT: " << baseRTT * 1e6 << " μs" << std::endl << std::endl;
    
    std::cout << "Baseline Parameters:" << std::endl;
    std::cout << "  α = " << baseline.alpha << std::endl;
    std::cout << "  β = " << baseline.beta << std::endl;
    std::cout << "  η = " << baseline.eta << std::endl << std::endl;
    
    // Check stability
    std::string msg;
    bool stable = baseline.checkStability(capacity, msg);
    std::cout << "Stability Check: " << msg << std::endl << std::endl;
    
    if (stable) {
        // Create flow
        HPCCFlow flow(baseline, capacity, baseRTT);
        
        // Simulate some telemetry updates
        std::cout << "Simulating telemetry feedback..." << std::endl;
        
        for (int i = 0; i < 10; i++) {
            LinkTelemetry telemetry;
            telemetry.qLen = 50000 * (1 + 0.1 * i);  // Increasing queue
            telemetry.txBytes = capacity * 0.01 / 8;
            telemetry.capacity = capacity;
            telemetry.timestamp = i * 0.01;
            
            flow.updateRate(telemetry, i > 0, nullptr);
            
            if (i % 3 == 0) {
                std::cout << "\nStep " << i << ":" << std::endl;
                flow.printState();
            }
        }
        
        std::cout << "\nFinal Metrics:" << std::endl;
        auto metrics = flow.getMetrics();
        std::cout << "  Avg Rate: " << metrics.avgRate / 1e9 << " Gbps" << std::endl;
        std::cout << "  Avg Utilization: " << metrics.avgUtilization << std::endl;
        std::cout << "  Avg Queue: " << metrics.avgQueue / 1024 << " KB" << std::endl;
        std::cout << "  Max Queue: " << metrics.maxQueue / 1024 << " KB" << std::endl;
    }
    
    return 0;
}

#endif // HPCC_TEST_MAIN
