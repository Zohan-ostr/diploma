#include <cmath>
#include <cstring>
#include <iostream>
#include <string>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#include "rclcpp/rclcpp.hpp"
#include "upper_body_msgs/msg/upper_body_command.hpp"

struct UdpArmPacket {
    uint32_t magic;
    uint32_t seq;
    double stamp_sec;
    float q[8]; // RSP, RSR, RSY, RE, LSP, LSR, LSY, LE
};

static constexpr uint32_t MAGIC = 0x48314152; // "H1AR"

class RosUpperBodyUdpForwarder : public rclcpp::Node {
public:
    RosUpperBodyUdpForwarder() : Node("ros_upper_body_udp_forwarder") {
        input_topic_ = declare_parameter<std::string>("input_topic", "/upper_body/command_geom");
        udp_host_ = declare_parameter<std::string>("udp_host", "127.0.0.1");
        udp_port_ = declare_parameter<int>("udp_port", 50051);

        sock_ = socket(AF_INET, SOCK_DGRAM, 0);
        if (sock_ < 0) {
            throw std::runtime_error("socket() failed");
        }

        std::memset(&addr_, 0, sizeof(addr_));
        addr_.sin_family = AF_INET;
        addr_.sin_port = htons(udp_port_);
        inet_pton(AF_INET, udp_host_.c_str(), &addr_.sin_addr);

        sub_ = create_subscription<upper_body_msgs::msg::UpperBodyCommand>(
            input_topic_,
            rclcpp::QoS(rclcpp::KeepLast(5)).reliable(),
            std::bind(&RosUpperBodyUdpForwarder::callback, this, std::placeholders::_1)
        );

        RCLCPP_INFO(get_logger(), "ROS UDP forwarder started");
        RCLCPP_INFO(get_logger(), "input_topic: %s", input_topic_.c_str());
        RCLCPP_INFO(get_logger(), "udp:         %s:%d", udp_host_.c_str(), udp_port_);
    }

    ~RosUpperBodyUdpForwarder() {
        if (sock_ >= 0) close(sock_);
    }

private:
    int index_for_name(const std::string& name) {
        if (name == "right_shoulder_pitch" || name == "right_shoulder_pitch_joint") return 0;
        if (name == "right_shoulder_roll"  || name == "right_shoulder_roll_joint")  return 1;
        if (name == "right_shoulder_yaw"   || name == "right_shoulder_yaw_joint")   return 2;
        if (name == "right_elbow"          || name == "right_elbow_joint")          return 3;
        if (name == "left_shoulder_pitch"  || name == "left_shoulder_pitch_joint")  return 4;
        if (name == "left_shoulder_roll"   || name == "left_shoulder_roll_joint")   return 5;
        if (name == "left_shoulder_yaw"    || name == "left_shoulder_yaw_joint")    return 6;
        if (name == "left_elbow"           || name == "left_elbow_joint")           return 7;
        return -1;
    }

    void callback(const upper_body_msgs::msg::UpperBodyCommand::SharedPtr msg) {
        if (!msg->valid) return;
        if (msg->joint_names.size() != msg->position.size()) return;

        UdpArmPacket pkt{};
        pkt.magic = MAGIC;
        pkt.seq = seq_++;
        pkt.stamp_sec = now().seconds();

        for (int i = 0; i < 8; ++i) {
            pkt.q[i] = NAN;
        }

        for (size_t i = 0; i < msg->joint_names.size(); ++i) {
            int idx = index_for_name(msg->joint_names[i]);
            if (idx >= 0 && std::isfinite(msg->position[i])) {
                pkt.q[idx] = static_cast<float>(msg->position[i]);
            }
        }

        sendto(sock_, &pkt, sizeof(pkt), 0, reinterpret_cast<sockaddr*>(&addr_), sizeof(addr_));

        RCLCPP_INFO_THROTTLE(
            get_logger(), *get_clock(), 1000,
            "udp seq=%u q=[%.3f %.3f %.3f %.3f | %.3f %.3f %.3f %.3f]",
            pkt.seq,
            pkt.q[0], pkt.q[1], pkt.q[2], pkt.q[3],
            pkt.q[4], pkt.q[5], pkt.q[6], pkt.q[7]
        );
    }

    std::string input_topic_;
    std::string udp_host_;
    int udp_port_;
    int sock_{-1};
    sockaddr_in addr_{};
    uint32_t seq_{0};
    rclcpp::Subscription<upper_body_msgs::msg::UpperBodyCommand>::SharedPtr sub_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<RosUpperBodyUdpForwarder>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
