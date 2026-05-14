#include <cmath>
#include <csignal>
#include <cstdint>
#include <iostream>
#include <thread>
#include <atomic>
#include <chrono>
#include <string>

#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/idl/go2/LowCmd_.hpp>
#include <unitree/idl/go2/LowState_.hpp>

using namespace unitree::robot;

#define TOPIC_LOWCMD   "rt/lowcmd"
#define TOPIC_LOWSTATE "rt/lowstate"

static constexpr int H1_NUM_MOTORS = 20;
static constexpr int RIGHT_SHOULDER_PITCH = 12;
static constexpr int RIGHT_SHOULDER_ROLL  = 13;
static constexpr int RIGHT_SHOULDER_YAW   = 14;
static constexpr int RIGHT_ELBOW          = 15;
static constexpr int LEFT_SHOULDER_PITCH  = 16;
static constexpr int LEFT_SHOULDER_ROLL   = 17;
static constexpr int LEFT_SHOULDER_YAW    = 18;
static constexpr int LEFT_ELBOW           = 19;

static constexpr float PosStopF = 2.146E+9f;
static constexpr float VelStopF = 16000.0f;

std::atomic_bool running{true};
unitree_go::msg::dds_::LowState_ g_low_state{};
std::atomic_bool g_lowstate_ok{false};

uint32_t crc32_core(uint32_t* ptr, uint32_t len)
{
    uint32_t xbit = 0;
    uint32_t data = 0;
    uint32_t CRC32 = 0xFFFFFFFF;
    const uint32_t dwPolynomial = 0x04c11db7;

    for (uint32_t i = 0; i < len; i++) {
        xbit = 1u << 31;
        data = ptr[i];
        for (uint32_t bits = 0; bits < 32; bits++) {
            if (CRC32 & 0x80000000) {
                CRC32 <<= 1;
                CRC32 ^= dwPolynomial;
            } else {
                CRC32 <<= 1;
            }
            if (data & xbit) CRC32 ^= dwPolynomial;
            xbit >>= 1;
        }
    }
    return CRC32;
}

bool is_arm_id(int id)
{
    return id == RIGHT_SHOULDER_PITCH || id == RIGHT_SHOULDER_ROLL ||
           id == RIGHT_SHOULDER_YAW   || id == RIGHT_ELBOW ||
           id == LEFT_SHOULDER_PITCH  || id == LEFT_SHOULDER_ROLL ||
           id == LEFT_SHOULDER_YAW    || id == LEFT_ELBOW;
}

void low_state_handler(const void* message)
{
    g_low_state = *(unitree_go::msg::dds_::LowState_*)message;
    g_lowstate_ok = true;
}

void signal_handler(int)
{
    running = false;
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <network_interface> [motor_id] [amplitude_rad] [kp] [kd] [duration_sec]\n";
        std::cerr << "Example: " << argv[0] << " eth0 12 0.10 20.0 1.0 6.0\n";
        return 1;
    }

    std::string iface = argv[1];
    int test_motor_id = (argc > 2) ? std::stoi(argv[2]) : RIGHT_SHOULDER_PITCH;
    float amplitude = (argc > 3) ? std::stof(argv[3]) : 0.10f;
    float kp = (argc > 4) ? std::stof(argv[4]) : 20.0f;
    float kd = (argc > 5) ? std::stof(argv[5]) : 1.0f;
    float duration_sec = (argc > 6) ? std::stof(argv[6]) : 6.0f;

    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    std::cout << "============================================================\n";
    std::cout << " H1 SDK2 RIGHT ARM TEST\n";
    std::cout << "============================================================\n";
    std::cout << "iface:          " << iface << "\n";
    std::cout << "topic cmd:      " << TOPIC_LOWCMD << "\n";
    std::cout << "topic state:    " << TOPIC_LOWSTATE << "\n";
    std::cout << "test_motor_id:  " << test_motor_id << "\n";
    std::cout << "amplitude_rad:  " << amplitude << "\n";
    std::cout << "kp:             " << kp << "\n";
    std::cout << "kd:             " << kd << "\n";
    std::cout << "duration_sec:   " << duration_sec << "\n";
    std::cout << "============================================================\n";
    std::cout << "WARNING: robot must be suspended/fixed. Type YES to start: ";

    std::string confirm;
    std::cin >> confirm;
    if (confirm != "YES") {
        std::cout << "Abort.\n";
        return 0;
    }

    ChannelFactory::Instance()->Init(0, iface);

    ChannelSubscriberPtr<unitree_go::msg::dds_::LowState_> lowstate_subscriber;
    lowstate_subscriber.reset(new ChannelSubscriber<unitree_go::msg::dds_::LowState_>(TOPIC_LOWSTATE));
    lowstate_subscriber->InitChannel(low_state_handler, 1);

    ChannelPublisherPtr<unitree_go::msg::dds_::LowCmd_> lowcmd_publisher;
    lowcmd_publisher.reset(new ChannelPublisher<unitree_go::msg::dds_::LowCmd_>(TOPIC_LOWCMD));
    lowcmd_publisher->InitChannel();

    std::cout << "Waiting for lowstate...\n";
    auto wait_start = std::chrono::steady_clock::now();
    while (!g_lowstate_ok && running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration<double>(now - wait_start).count() > 5.0) {
            std::cerr << "FAIL: no lowstate. Check iface/dev mode.\n";
            return 2;
        }
    }

    float base_q[H1_NUM_MOTORS]{};
    for (int i = 0; i < H1_NUM_MOTORS; ++i) base_q[i] = g_low_state.motor_state()[i].q();

    std::cout << "Lowstate OK\n";
    std::cout << "Initial arm q: [";
    for (int i = 12; i <= 19; ++i) std::cout << base_q[i] << (i == 19 ? "" : ", ");
    std::cout << "]\n";

    unitree_go::msg::dds_::LowCmd_ cmd{};
    cmd.head()[0] = 0xFE;
    cmd.head()[1] = 0xEF;
    cmd.level_flag() = 0xFF;
    cmd.gpio() = 0;

    const float rate_hz = 250.0f;
    const float dt = 1.0f / rate_hz;
    const float period_sec = 4.0f;
    auto start = std::chrono::steady_clock::now();
    double last_print = -1.0;

    while (running) {
        auto now = std::chrono::steady_clock::now();
        double t = std::chrono::duration<double>(now - start).count();

        float target[H1_NUM_MOTORS]{};
        for (int i = 0; i < H1_NUM_MOTORS; ++i) target[i] = base_q[i];

        if (t <= duration_sec) {
            target[test_motor_id] = base_q[test_motor_id] + amplitude * std::sin(2.0 * M_PI * t / period_sec);
        } else if (t <= duration_sec + 2.0) {
            target[test_motor_id] = base_q[test_motor_id];
        } else {
            break;
        }

        for (int i = 0; i < H1_NUM_MOTORS; ++i) {
            auto& m = cmd.motor_cmd()[i];
            if (is_arm_id(i)) {
                m.mode() = 0x01;
                m.q() = target[i];
                m.dq() = 0.0f;
                m.kp() = kp;
                m.kd() = kd;
                m.tau() = 0.0f;
            } else {
                m.mode() = 0x01;
                m.q() = PosStopF;
                m.dq() = VelStopF;
                m.kp() = 0.0f;
                m.kd() = 0.0f;
                m.tau() = 0.0f;
            }
        }

        cmd.crc() = crc32_core((uint32_t *)&cmd, (sizeof(unitree_go::msg::dds_::LowCmd_) >> 2) - 1);
        lowcmd_publisher->Write(cmd);

        if (t - last_print >= 0.5) {
            float current = g_low_state.motor_state()[test_motor_id].q();
            std::cout << "t=" << t << " target=" << target[test_motor_id]
                      << " current=" << current << " crc=" << cmd.crc() << "\n";
            last_print = t;
        }

        std::this_thread::sleep_for(std::chrono::duration<float>(dt));
    }

    std::cout << "Done.\n";
    return 0;
}
