#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <thread>
#include <atomic>
#include <chrono>
#include <mutex>
#include <string>
#include <fcntl.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/idl/go2/LowCmd_.hpp>
#include <unitree/idl/go2/LowState_.hpp>

using namespace unitree::robot;

#define TOPIC_LOWCMD   "rt/lowcmd"
#define TOPIC_LOWSTATE "rt/lowstate"

static constexpr int H1_NUM_MOTORS = 20;

static constexpr int RSP = 12;
static constexpr int RSR = 13;
static constexpr int RSY = 14;
static constexpr int RE  = 15;
static constexpr int LSP = 16;
static constexpr int LSR = 17;
static constexpr int LSY = 18;
static constexpr int LE  = 19;

static constexpr int ARM_IDS[8] = {RSP, RSR, RSY, RE, LSP, LSR, LSY, LE};

static constexpr float PosStopF = 2.146E+9f;
static constexpr float VelStopF = 16000.0f;

static constexpr uint32_t MAGIC = 0x48314152; // "H1AR"

struct UdpArmPacket {
    uint32_t magic;
    uint32_t seq;
    double stamp_sec;
    float q[8]; // RSP, RSR, RSY, RE, LSP, LSR, LSY, LE
};

std::atomic_bool running{true};
std::atomic_bool g_lowstate_ok{false};
unitree_go::msg::dds_::LowState_ g_low_state{};
std::mutex g_lowstate_mutex;

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

            if (data & xbit) {
                CRC32 ^= dwPolynomial;
            }

            xbit >>= 1;
        }
    }

    return CRC32;
}

bool is_arm_id(int id)
{
    return id >= 12 && id <= 19;
}

void low_state_handler(const void* message)
{
    std::lock_guard<std::mutex> lock(g_lowstate_mutex);
    g_low_state = *(unitree_go::msg::dds_::LowState_*)message;
    g_lowstate_ok = true;
}

void signal_handler(int)
{
    running = false;
}

double now_sec()
{
    using clock = std::chrono::steady_clock;
    static auto t0 = clock::now();
    auto t = clock::now();
    return std::chrono::duration<double>(t - t0).count();
}

int main(int argc, char** argv)
{
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    std::string iface = "eth0";
    int domain = 0;
    int udp_port = 50051;
    float kp = 25.0f;
    float kd = 1.5f;
    float max_step = 0.012f;
    float timeout_sec = 0.35f;

    if (argc > 1) iface = argv[1];
    if (argc > 2) kp = std::stof(argv[2]);
    if (argc > 3) kd = std::stof(argv[3]);
    if (argc > 4) max_step = std::stof(argv[4]);
    if (argc > 5) timeout_sec = std::stof(argv[5]);
    if (argc > 6) udp_port = std::stoi(argv[6]);

    std::cout << "============================================================\n";
    std::cout << " SDK2 H1 UDP LOWCMD SENDER\n";
    std::cout << "============================================================\n";
    std::cout << "iface:       " << iface << "\n";
    std::cout << "domain:      " << domain << "\n";
    std::cout << "udp_port:    " << udp_port << "\n";
    std::cout << "kp:          " << kp << "\n";
    std::cout << "kd:          " << kd << "\n";
    std::cout << "max_step:    " << max_step << "\n";
    std::cout << "timeout_sec: " << timeout_sec << "\n";
    std::cout << "============================================================\n";
    std::cout << "WARNING: this sends commands to rt/lowcmd. Type YES to start: ";

    std::string confirm;
    std::cin >> confirm;
    if (confirm != "YES") {
        std::cout << "Abort.\n";
        return 0;
    }

    int udp_sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (udp_sock < 0) {
        std::cerr << "socket failed\n";
        return 1;
    }

    sockaddr_in bind_addr{};
    bind_addr.sin_family = AF_INET;
    bind_addr.sin_addr.s_addr = htonl(INADDR_ANY);  // accept UDP from laptop and localhost
    bind_addr.sin_port = htons(udp_port);

    if (bind(udp_sock, reinterpret_cast<sockaddr*>(&bind_addr), sizeof(bind_addr)) < 0) {
        std::cerr << "bind UDP failed\n";
        return 1;
    }

    int flags = fcntl(udp_sock, F_GETFL, 0);
    fcntl(udp_sock, F_SETFL, flags | O_NONBLOCK);

    ChannelFactory::Instance()->Init(domain, iface);

    ChannelSubscriberPtr<unitree_go::msg::dds_::LowState_> lowstate_subscriber;
    lowstate_subscriber.reset(new ChannelSubscriber<unitree_go::msg::dds_::LowState_>(TOPIC_LOWSTATE));
    lowstate_subscriber->InitChannel(low_state_handler, 1);

    ChannelPublisherPtr<unitree_go::msg::dds_::LowCmd_> lowcmd_publisher;
    lowcmd_publisher.reset(new ChannelPublisher<unitree_go::msg::dds_::LowCmd_>(TOPIC_LOWCMD));
    lowcmd_publisher->InitChannel();

    std::cout << "Waiting for rt/lowstate...\n";
    auto wait_start = std::chrono::steady_clock::now();

    while (!g_lowstate_ok && running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        auto t = std::chrono::steady_clock::now();
        if (std::chrono::duration<double>(t - wait_start).count() > 5.0) {
            std::cerr << "FAIL: no rt/lowstate\n";
            return 2;
        }
    }

    float target_q[H1_NUM_MOTORS]{};
    float sent_q[H1_NUM_MOTORS]{};
    float base_q[H1_NUM_MOTORS]{};

    {
        std::lock_guard<std::mutex> lock(g_lowstate_mutex);
        for (int i = 0; i < H1_NUM_MOTORS; ++i) {
            base_q[i] = g_low_state.motor_state()[i].q();
            target_q[i] = base_q[i];
            sent_q[i] = base_q[i];
        }
    }

    std::cout << "rt/lowstate OK\n";
    std::cout << "Initial arm q: [";
    for (int i = 12; i <= 19; ++i) {
        std::cout << base_q[i] << (i == 19 ? "" : ", ");
    }
    std::cout << "]\n";

    unitree_go::msg::dds_::LowCmd_ cmd{};
    cmd.head()[0] = 0xFE;
    cmd.head()[1] = 0xEF;
    cmd.level_flag() = 0xFF;
    cmd.gpio() = 0;

    double last_udp_time = -1000.0;
    double last_print = -1000.0;
    uint32_t last_seq = 0;

    const float rate_hz = 250.0f;
    const float dt = 1.0f / rate_hz;

    while (running) {
        UdpArmPacket pkt{};
        while (true) {
            ssize_t n = recv(udp_sock, &pkt, sizeof(pkt), 0);
            if (n != sizeof(pkt)) break;
            if (pkt.magic != MAGIC) continue;

            for (int i = 0; i < 8; ++i) {
                if (std::isfinite(pkt.q[i])) {
                    target_q[ARM_IDS[i]] = pkt.q[i];
                }
            }

            last_udp_time = now_sec();
            last_seq = pkt.seq;
        }

        bool timeout = (now_sec() - last_udp_time) > timeout_sec;

        if (timeout) {
            for (int id : ARM_IDS) {
                target_q[id] = sent_q[id];
            }
        }

        for (int i = 0; i < H1_NUM_MOTORS; ++i) {
            auto& m = cmd.motor_cmd()[i];

            if (is_arm_id(i)) {
                float delta = target_q[i] - sent_q[i];
                if (delta > max_step) delta = max_step;
                if (delta < -max_step) delta = -max_step;

                sent_q[i] += delta;

                m.mode() = 0x01;
                m.q() = sent_q[i];
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

        double t = now_sec();
        if (t - last_print > 0.5) {
            float cur[H1_NUM_MOTORS]{};
            {
                std::lock_guard<std::mutex> lock(g_lowstate_mutex);
                for (int i = 0; i < H1_NUM_MOTORS; ++i) {
                    cur[i] = g_low_state.motor_state()[i].q();
                }
            }

            std::cout
                << "timeout=" << (timeout ? 1 : 0)
                << " seq=" << last_seq
                << " target_r=[" << target_q[12] << " " << target_q[13] << " " << target_q[14] << " " << target_q[15] << "]"
                << " sent_r=[" << sent_q[12] << " " << sent_q[13] << " " << sent_q[14] << " " << sent_q[15] << "]"
                << " current_r=[" << cur[12] << " " << cur[13] << " " << cur[14] << " " << cur[15] << "]"
                << "\n";

            last_print = t;
        }

        std::this_thread::sleep_for(std::chrono::duration<float>(dt));
    }

    close(udp_sock);
    std::cout << "Done.\n";
    return 0;
}
