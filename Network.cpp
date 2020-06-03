#include "cxxopts.hpp"
#include <iostream>
#include <algorithm>
#include <string>
#include <thread>
#include <openvr.h>
#include <vrinputemulator.h>
#include <vector>
#define GLM_FORCE_SWIZZLE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp> 
#include <glm/gtx/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <csignal>

#include <fstream>
#include <cmath>

#ifdef _WIN32
#include <Windows.h>
#endif

#define TRACKER_ADJUSTMENT_SYSTEM "v0.0.1"

/// VR System
static vr::IVRSystem* m_VRSystem;
static vrinputemulator::VRInputEmulator inputEmulator;
/// Time Domain
// Time since last frame in seconds.
static float deltaTime;
// Current Frame information
static int currentFrame;

/// Tracker
static vr::TrackedDevicePose_t devicePoses[vr::k_unMaxTrackedDeviceCount];
// Stores the positions of each device for the current frame.
static glm::vec3 devicePos[vr::k_unMaxTrackedDeviceCount];
// Stores the quaternions of each device for the current frame.
static glm::quat deviceRot[vr::k_unMaxTrackedDeviceCount];
// Fake Pelvis Tracker ID
static uint32_t pelvisTrackerID;

/// Connect Base stations, then controllers, then foot trackers, then pelvis tracker
// Real Pelvis Tracker ID
static uint32_t realPelvisTrackerID = 7;

/// Log
// logFile Location
std::ofstream logFile("log.txt");

/// Network
// Network Parameters
constexpr int N_INPUT = 35;
constexpr int N_HIDDEN1 = 50;
constexpr int N_HIDDEN2 = 50;
constexpr int N_OUTPUT = 7;

// Evaluate interval to decide network to train or not
constexpr int EVALUATE_INTERVAL = 1000;
// Training interval to limit over training
constexpr int TRAINING_INTERVAL = 5;
// Threshold of error to choose network to train or not
constexpr float THRESHOLD_ERROR = 1.0f;
// Number of frames from start
static int nFrames = 0;
// Number of frames that every devices are working well
static int nFramesStable = 0;
// Variable to hold error
static float error = 0;
//static float error[EVALUATE_INTERVAL];
// Variable to check if network is trainig or not
static bool isNetworkTraining = true;

// time
clock_t timeD = clock();
float timeT = 0.0;

#define M_E            2.7182818284590452354

// Starts training when data is valid
// When training loss of recent 100 data is below certain level, stops training
// Restart training when training loss increases again
static class Network {
public:
	// Weights
	float l1_w[N_INPUT][N_HIDDEN1];
	float l2_w[N_HIDDEN1][N_HIDDEN2];
	float l3_w[N_HIDDEN2][N_OUTPUT];
	// Values
	float i_v[N_INPUT];
	float l1_v[N_HIDDEN1];
	float l2_v[N_HIDDEN2];
	float l3_v[N_OUTPUT];
	float o_v[N_OUTPUT];
	// Gradients
	float l1_g[N_HIDDEN1];
	float l2_g[N_HIDDEN2];
	float l3_g[N_OUTPUT];
	// Other configs
	const int IDIndex[5] = { 0, 3, 4, 5, 6 };
	const int N_DEVICES = 5;
	float avg[3];
	float lr = 0.005f;

	Network() {
		srand(time(NULL));
		float range = 1;
		for (int ii = 0; ii < N_INPUT; ii++) {
			for (int io = 0; io < N_HIDDEN1; io++) {
				l1_w[ii][io] = rand() / (float)RAND_MAX * 2 * range - range;
			}
		}
		for (int ii = 0; ii < N_HIDDEN1; ii++) {
			for (int io = 0; io < N_HIDDEN2; io++) {
				l2_w[ii][io] = rand() / (float)RAND_MAX * 2 * range - range;
			}
		}
		for (int ii = 0; ii < N_HIDDEN2; ii++) {
			for (int io = 0; io < N_OUTPUT; io++) {
				l3_w[ii][io] = rand() / (float)RAND_MAX * 2 * range - range;
			}
		}
		printf("Network Initialized with random values\n");
	}

	bool IsInputDevice(int ind) {
		for (int i = 0; i < N_DEVICES; i++) {
			if (IDIndex[i] == ind) {
				return true;
			}
		}
		return false;
	}
	int GetNetworkIndex(int ind) {
		for (int i = 0; i < N_DEVICES; i++) {
			if (IDIndex[i] == ind) {
				return i;
			}
		}
		return -1;
	}
	void SetInputIndex(int ind, glm::vec3 p, glm::quat q) {
		int mind = GetNetworkIndex(ind);
		i_v[(mind * 7) + 0] = p.x;
		i_v[(mind * 7) + 1] = p.y;
		i_v[(mind * 7) + 2] = p.z;
		i_v[(mind * 7) + 3] = q.x;
		i_v[(mind * 7) + 4] = q.y;
		i_v[(mind * 7) + 5] = q.z;
		i_v[(mind * 7) + 6] = q.w;
	}

	// Activation function works kinda weird...
	float BackActivation(float v) {
		// v = (1 - v) * v;
		if (v > 4) { return 4.0f; }
		else if (v < -4) { return -4.0f; }
		return v;
	}

	float Activation(float v) {
		if (v > 4) { return 4.0f; }
		else if (v < -4) { return -4.0f; }
		return v;
		/*
		return 1.0 / (1 + pow(M_E, v)) - 1.0;
		*/
		/*
		if (l2_v[io] < 0.0f) { l2_v[io] = 0.0f; }
		else if (l2_v[io] > 6.0f) { l2_v[io] = 6.0f; }
		*/
	}


	void Print() {

		printf("\n\nNetwork Information on frame %d\n", nFrames);

		printf("Node Values\n");
		printf("Input\n");
		for (int i = 0; i < N_INPUT  ; i++) { printf("%.2f ", i_v[i]); }
		printf("\n");
		printf("L1\n");
		for (int i = 0; i < N_HIDDEN1; i++) { printf("%.2f ", l1_v[i]); }
		printf("\n");
		printf("L2\n");
		for (int i = 0; i < N_HIDDEN2; i++) { printf("%.2f ", l2_v[i]); }
		printf("\n");
		printf("L3\n");
		for (int i = 0; i < N_OUTPUT ; i++) { printf("%.2f ", l3_v[i]); }
		printf("\n");
		printf("Output processed ");
		for (int i = 0; i < N_OUTPUT ; i++) { printf("%.2f ", o_v[i]); }
		printf("\n"); printf("\n");

		printf("Gradients\n");
		printf("L3\n");
		for (int i = 0; i < N_OUTPUT ; i++) { printf("%.2f ", l3_g[i]); }
		printf("\n");
		printf("L2\n");
		for (int i = 0; i < N_HIDDEN2; i++) { printf("%.2f ", l2_g[i]); }
		printf("\n");
		printf("L1\n");
		for (int i = 0; i < N_HIDDEN1; i++) { printf("%.2f ", l1_g[i]); }
		printf("\n"); printf("\n");

		
		printf("Weights of First node\n");
		printf("L1\n");
		for (int i = 0; i < N_HIDDEN1; i++) { printf("%.2f ", l1_w[0][i]); }
		printf("\n");
		printf("L2\n");
		for (int i = 0; i < N_HIDDEN2; i++) { printf("%.2f ", l2_w[0][i]); }
		printf("\n");
		printf("L3\n");
		for (int i = 0; i < N_OUTPUT; i++) { printf("%.2f ", l3_w[0][i]); }
		printf("\n");
	}

	void Forward() {
		// Process input using devicePos and deviceRot
		// Get average position of values and adjust using that values, normalize
		avg[0] = 0.0f; avg[1] = 0.0f; avg[2] = 0.0f;
		for (int i = 0; i < N_DEVICES; i++) {
			avg[0] += devicePos[IDIndex[i]].x;
			avg[1] += devicePos[IDIndex[i]].y;
			avg[2] += devicePos[IDIndex[i]].z;
		}
		avg[0] /= N_DEVICES; avg[1] /= N_DEVICES; avg[2] /= N_DEVICES;
		for (int i = 0; i < N_DEVICES; i++) {
			i_v[(i + 1) * 7 + 0] += devicePos[IDIndex[i]].x - avg[0];
			i_v[(i + 1) * 7 + 1] += devicePos[IDIndex[i]].y - avg[1];
			i_v[(i + 1) * 7 + 2] += devicePos[IDIndex[i]].z - avg[2];
		}

		// Do I have to process additional with rotation? - Let's try to use basic values w/o processing
		for (int i = 0; i < N_DEVICES; i++) {
			i_v[(i + 1) * 7 + 3] = deviceRot[IDIndex[i]].x;
			i_v[(i + 1) * 7 + 4] = deviceRot[IDIndex[i]].y;
			i_v[(i + 1) * 7 + 5] = deviceRot[IDIndex[i]].z;
			i_v[(i + 1) * 7 + 6] = deviceRot[IDIndex[i]].w;
		}

		// Layer 1
		for (int io = 0; io < N_HIDDEN1; io++) {
			l1_v[io] = 0.0f;
			for (int ii = 0; ii < N_INPUT; ii++) {
				l1_v[io] += i_v[ii] * l1_w[ii][io];
			}
			// Activation
			l1_v[io] = Activation(l1_v[io]);
		}
		// Layer 2
		for (int io = 0; io < N_HIDDEN2; io++) {
			l2_v[io] = 0.0f;
			for (int ii = 0; ii < N_HIDDEN1; ii++) {
				l2_v[io] += l1_v[ii] * l2_w[ii][io];
			}
			// Activation
			l2_v[io] = Activation(l2_v[io]);
		}
		// Layer 3
		for (int io = 0; io < N_OUTPUT; io++) {
			l3_v[io] = 0.0f;
			for (int ii = 0; ii < N_HIDDEN2; ii++) {
				l3_v[io] += l2_v[ii] * l3_w[ii][io];
			}
		}

		// Re-process results
		for (int i = 0; i < 3; i++) {
			o_v[i] = l3_v[i] + avg[i];
		}
		for (int i = 3; i < 7; i++) {
			o_v[i] = l3_v[i];
		}
	}
	void Backward(float* answer) {

		// Convert locations
		for (int i = 0; i < 3; i++) {
			l3_g[i] = l3_v[i] - (answer[i] - avg[i]);
			//l3_g[i] = l3_v[i] - answer[i];
		}
		for (int i = 3; i < 7; i++) {
			l3_g[i] = l3_v[i] - answer[i];
		}


		// Backward propagation of layer 3
		for (int ii = 0; ii < N_HIDDEN2; ii++) {
			for (int io = 0; io < N_OUTPUT; io++) {
				l3_w[ii][io] -= l3_g[io] * l2_v[ii] * lr;
			}
		}

		// Calculate layer 2's gradient
		for (int ii = 0; ii < N_HIDDEN2; ii++) {
			float d = 0.0;
			for (int io = 0; io < N_OUTPUT; io++) {
				d += l2_w[ii][io] * l3_g[io];
			}
			l2_g[ii] = BackActivation(d);
		}

		// Backward propagation of layer 2
		for (int ii = 0; ii < N_HIDDEN1; ii++) {
			for (int io = 0; io < N_HIDDEN2; io++) {
				l2_w[ii][io] -= l2_g[io] * l1_v[ii] * lr;
			}
		}

		// Calculate layer 1's gradient
		for (int ii = 0; ii < N_HIDDEN1; ii++) {
			float d = 0.0;
			for (int io = 0; io < N_HIDDEN2; io++) {
				d += l1_w[ii][io] * l2_g[io];
			}
			l1_g[ii] = BackActivation(d);
		}



		// Backward propagation of layer 1
		for (int ii = 0; ii < N_INPUT; ii++) {
			for (int io = 0; io < N_HIDDEN1; io++) {
				l1_w[ii][io] -= l1_g[io] * i_v[ii] * lr;
			}
		}
	}
};

Network network;


// Create Fake Tracker
uint32_t createTracker() {
	uint32_t id = inputEmulator.getVirtualDeviceCount();
	//uint32_t id = 12;
	inputEmulator.addVirtualDevice(vrinputemulator::VirtualDeviceType::TrackedController, std::to_string(id), false);
	inputEmulator.setVirtualDeviceProperty(id, vr::ETrackedDeviceProperty::Prop_TrackingSystemName_String, "lighthouse");
	inputEmulator.setVirtualDeviceProperty(id, vr::ETrackedDeviceProperty::Prop_ModelNumber_String, "Vive Tracker Pro MV");
	inputEmulator.setVirtualDeviceProperty(id, vr::ETrackedDeviceProperty::Prop_RenderModelName_String, "{htc}vr_tracker_vive_1_0");
	inputEmulator.setVirtualDeviceProperty(id, vr::ETrackedDeviceProperty::Prop_WillDriftInYaw_Bool, false);
	inputEmulator.setVirtualDeviceProperty(id, vr::ETrackedDeviceProperty::Prop_ManufacturerName_String, "HTC");
	inputEmulator.setVirtualDeviceProperty(id, vr::ETrackedDeviceProperty::Prop_TrackingFirmwareVersion_String, "1541806442 RUNNER-WATCHMAN$runner-watchman@runner-watchman 2018-11-10 FPGA 531(2.19/7/2) BL 0 VRC 1541806442 Radio 1518811657");
	inputEmulator.setVirtualDeviceProperty(id, vr::ETrackedDeviceProperty::Prop_HardwareRevision_String, "product 132 rev 2.0.6 lot 2000/0/0 0");
	inputEmulator.setVirtualDeviceProperty(id, vr::ETrackedDeviceProperty::Prop_DeviceIsWireless_Bool, true);
	inputEmulator.setVirtualDeviceProperty(id, vr::ETrackedDeviceProperty::Prop_HardwareRevision_Uint64, (uint64_t)2214723590);
	inputEmulator.setVirtualDeviceProperty(id, vr::ETrackedDeviceProperty::Prop_FirmwareVersion_Uint64, (uint64_t)1541806442);
	inputEmulator.setVirtualDeviceProperty(id, vr::ETrackedDeviceProperty::Prop_DeviceClass_Int32, 3);
	inputEmulator.setVirtualDeviceProperty(id, vr::ETrackedDeviceProperty::Prop_SupportedButtons_Uint64, (uint64_t)12884901895);
	inputEmulator.setVirtualDeviceProperty(id, vr::ETrackedDeviceProperty::Prop_Axis0Type_Int32, 0);
	inputEmulator.setVirtualDeviceProperty(id, vr::ETrackedDeviceProperty::Prop_Axis1Type_Int32, 0);
	inputEmulator.setVirtualDeviceProperty(id, vr::ETrackedDeviceProperty::Prop_Axis2Type_Int32, 0);
	inputEmulator.setVirtualDeviceProperty(id, vr::ETrackedDeviceProperty::Prop_Axis3Type_Int32, 0);
	inputEmulator.setVirtualDeviceProperty(id, vr::ETrackedDeviceProperty::Prop_Axis4Type_Int32, 0);
	inputEmulator.setVirtualDeviceProperty(id, vr::ETrackedDeviceProperty::Prop_IconPathName_String, "icons");
	inputEmulator.setVirtualDeviceProperty(id, vr::ETrackedDeviceProperty::Prop_NamedIconPathDeviceOff_String, "{htc}tracker_status_off.png");
	inputEmulator.setVirtualDeviceProperty(id, vr::ETrackedDeviceProperty::Prop_NamedIconPathDeviceSearching_String, "{htc}tracker_status_searching.gif");
	inputEmulator.setVirtualDeviceProperty(id, vr::ETrackedDeviceProperty::Prop_NamedIconPathDeviceSearchingAlert_String, "{htc}tracker_status_alert.gif");
	inputEmulator.setVirtualDeviceProperty(id, vr::ETrackedDeviceProperty::Prop_NamedIconPathDeviceReady_String, "{htc}tracker_status_ready.png");
	inputEmulator.setVirtualDeviceProperty(id, vr::ETrackedDeviceProperty::Prop_NamedIconPathDeviceNotReady_String, "{htc}tracker_status_error.png");
	inputEmulator.setVirtualDeviceProperty(id, vr::ETrackedDeviceProperty::Prop_NamedIconPathDeviceStandby_String, "{htc}tracker_status_standby.png");
	inputEmulator.setVirtualDeviceProperty(id, vr::ETrackedDeviceProperty::Prop_NamedIconPathDeviceAlertLow_String, "{htc}tracker_status_ready_low.png");
	inputEmulator.publishVirtualDevice(id);
	return id;
}

// Delete Fake Tracker
void deleteVirtualDevice(int id) {
	vr::DriverPose_t pose = inputEmulator.getVirtualDevicePose(id);
	pose.deviceIsConnected = false;
	pose.result = vr::TrackingResult_Uninitialized;
	pose.poseIsValid = false;
	inputEmulator.setVirtualDevicePose(id, pose, false);
}

// On close, close handler
void onClose() {
	deleteVirtualDevice(pelvisTrackerID);
	inputEmulator.disconnect();
}
void signalHandler(int signum) {
	std::cerr << "Interrupt signal (" << signum << ") received, cleaning up...\n" << std::flush;
	onClose();
	exit(signum);
}
#ifdef _WIN32
BOOL WINAPI ConsoleHandler(DWORD CEvent) {
	std::cerr << "Console window was closed, cleaning up...\n" << std::flush;
	if (CEvent == CTRL_CLOSE_EVENT) {
		onClose();
	}
	return TRUE;
}
#endif


glm::vec3 GetPosition(vr::HmdMatrix34_t matrix) {
	glm::vec3 p;
	p.x = matrix.m[0][3];
	p.y = matrix.m[1][3];
	p.z = matrix.m[2][3];
	return p;
}
glm::quat GetRotation(vr::HmdMatrix34_t matrix) {
	glm::quat q;

	q.w = sqrt(fmax(0, 1 + matrix.m[0][0] + matrix.m[1][1] + matrix.m[2][2])) / 2;
	q.x = sqrt(fmax(0, 1 + matrix.m[0][0] - matrix.m[1][1] - matrix.m[2][2])) / 2;
	q.y = sqrt(fmax(0, 1 - matrix.m[0][0] + matrix.m[1][1] - matrix.m[2][2])) / 2;
	q.z = sqrt(fmax(0, 1 - matrix.m[0][0] - matrix.m[1][1] + matrix.m[2][2])) / 2;
	q.x = copysign(q.x, matrix.m[2][1] - matrix.m[1][2]);
	q.y = copysign(q.y, matrix.m[0][2] - matrix.m[2][0]);
	q.z = copysign(q.z, matrix.m[1][0] - matrix.m[0][1]);
	return q;
}

void setVirtualDevicePosition(uint32_t id, glm::vec3 pos, glm::quat rot) {
	vr::DriverPose_t pose = inputEmulator.getVirtualDevicePose(id);
	pose.vecPosition[0] = pos.x;
	pose.vecPosition[1] = pos.y;
	pose.vecPosition[2] = pos.z;
	pose.poseIsValid = true;
	pose.deviceIsConnected = true;
	pose.result = vr::TrackingResult_Running_OK;
	pose.qRotation.w = rot.w;
	pose.qRotation.x = rot.x;
	pose.qRotation.y = rot.y;
	pose.qRotation.z = rot.z;
	inputEmulator.setVirtualDevicePose(id, pose, false);
}


void Update() {
	float fSecondsSinceLastVsync;
	vr::VRSystem()->GetTimeSinceLastVsync(&fSecondsSinceLastVsync, NULL);
	float fDisplayFrequency = vr::VRSystem()->GetFloatTrackedDeviceProperty(vr::k_unTrackedDeviceIndex_Hmd, vr::Prop_DisplayFrequency_Float);
	float fFrameDuration = 1.f / fDisplayFrequency;
	float fVsyncToPhotons = vr::VRSystem()->GetFloatTrackedDeviceProperty(vr::k_unTrackedDeviceIndex_Hmd, vr::Prop_SecondsFromVsyncToPhotons_Float);
	float fPredictedSecondsFromNow = fFrameDuration - fSecondsSinceLastVsync + fVsyncToPhotons;
	vr::VRSystem()->GetDeviceToAbsoluteTrackingPose(vr::TrackingUniverseRawAndUncalibrated, fPredictedSecondsFromNow, devicePoses, vr::k_unMaxTrackedDeviceCount);

	for (uint32_t deviceIndex = 0; deviceIndex < vr::k_unMaxTrackedDeviceCount; deviceIndex++) {
		if (!vr::VRSystem()->IsTrackedDeviceConnected(deviceIndex)) { // Continue if device is not connected
			continue;
		}
		vr::TrackedDevicePose_t* pose = devicePoses + deviceIndex;

		// Update device position
		if (pose->bPoseIsValid && pose->bDeviceIsConnected) {
			// deviceLastPos[deviceIndex] = devicePos[deviceIndex];
			glm::vec3 p = GetPosition(pose->mDeviceToAbsoluteTracking);
			glm::quat q = GetRotation(pose->mDeviceToAbsoluteTracking);
			devicePos[deviceIndex] = p;
			deviceRot[deviceIndex] = q;
		}
		
	}

	timeD = clock();

	bool isValid = true;
	// Check data is valid or not
	for (uint32_t deviceIndex = 0; deviceIndex < 8; deviceIndex++) {
		if (!vr::VRSystem()->IsTrackedDeviceConnected(deviceIndex)) {
			isValid = false;
			break;
		}
	}

	nFrames++;

	if (isValid) {
		nFramesStable++;

		// Forward the network and get the error 
		network.Forward();

		// Get error
		vr::TrackedDevicePose_t* pose = devicePoses + realPelvisTrackerID;
		glm::vec3 rp = GetPosition(pose->mDeviceToAbsoluteTracking);
		glm::quat rq = GetRotation(pose->mDeviceToAbsoluteTracking);
		float errors[7];
		errors[0] = (network.o_v[0] - rp.x);
		errors[1] = (network.o_v[1] - rp.y);
		errors[2] = (network.o_v[2] - rp.z);
		errors[3] = (network.o_v[3] - rq.x);
		errors[4] = (network.o_v[4] - rq.y);
		errors[5] = (network.o_v[5] - rq.z);
		errors[6] = (network.o_v[6] - rq.w);

		// Add error of the network
		float et = 0.0f;
		for (int i = 0; i < 7; i++) {
			et += errors[i]*errors[i];
		}
		error += sqrt(et);

		if (nFramesStable % 1000 == 10) {
			//network.Print();
			printf("Errors: ");
			for (int i = 0; i < 7; i++) {
				printf("%.2f ", errors[i]);
			}
			printf("\n");
			printf("Original Value: %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n",rp.x, rp.y, rp.z, rq.x, rq.y, rq.z, rq.w);

		}

		// If network is in training mode, train the network.
		if (nFramesStable % TRAINING_INTERVAL == 0 && isNetworkTraining) {
			//printf("Backward!\n");
			network.Backward(errors);
			// network.Print();
		}

		if (nFramesStable % EVALUATE_INTERVAL == 0) {
			float e = error / float(EVALUATE_INTERVAL);
			printf("Error : %.3f\n", e);
			if (e < THRESHOLD_ERROR) {
				if (isNetworkTraining){
					isNetworkTraining = false;
					printf("Training Disbled\n");
				}
			}
			else{
				if (!isNetworkTraining) {
					isNetworkTraining = true;
					printf("Training Enabled\n");
				}
			}
			error = 0.0f;
		}

	}
	else { // If tracker has error
		// Forward to the network
		network.Forward();
		// Update fake tracker
		glm::vec3 p;
		p.x = network.o_v[0];	p.y = network.o_v[1];	p.z = network.o_v[2];
		glm::quat q;
		q.x = network.o_v[3];	q.y = network.o_v[4];	q.z = network.o_v[5];	q.w = network.o_v[6];
		setVirtualDevicePosition(pelvisTrackerID, p, q);
		printf("Updated tracker position!");
	}

	if (nFrames % 1000 == 10) {
		printf("%d Frames Processed, %d valid frames\n", nFrames, nFramesStable);
		printf("%f time\n", timeT);
	}
	timeT += float(clock() - timeD) / CLOCKS_PER_SEC;
	/* Try to disabling original tracker..?
	vr::DriverPose_t p;
	p.deviceIsConnected = false;
	inputEmulator.openvrUpdatePose(realPelvisTrackerID, p);
	inputEmulator.setDeviceRedictMode(realPelvisTrackerID, true);*/
	//inputEmulator.setDeviceFakeDisconnectedMode(realPelvisTrackerID, true);

}



void logWriteToFile() {
	for (uint32_t deviceIndex = 0; deviceIndex < vr::k_unMaxTrackedDeviceCount; deviceIndex++) {
		if (!vr::VRSystem()->IsTrackedDeviceConnected(deviceIndex)) { continue; }
		logFile << deviceIndex << "\t";
		if (devicePoses[deviceIndex].eTrackingResult == 200) { logFile << "1\t"; }
		else { logFile << "0\t"; }

		logFile << devicePos[deviceIndex].x << "\t" << devicePos[deviceIndex].y << "\t" << devicePos[deviceIndex].z << "\t";
		logFile << deviceRot[deviceIndex].x << "\t" << deviceRot[deviceIndex].y << "\t" << deviceRot[deviceIndex].z << "\t" << deviceRot[deviceIndex].w << "\t";

		//std::cout << deviceIndex << ":" << devicePoses[deviceIndex].eTrackingResult << ", " ; // 200 = normal, 101 = error
		//printf("%d:(%.2f,%.2f,%.2f,%.2f) ", deviceIndex, deviceRot[deviceIndex].x, deviceRot[deviceIndex].y, deviceRot[deviceIndex].z, deviceRot[deviceIndex].w);
	}
	logFile << "\n";
}

void printTrackerInfo() {
	for (uint32_t deviceIndex = 0; deviceIndex < vr::k_unMaxTrackedDeviceCount; deviceIndex++) {
		if (!vr::VRSystem()->IsTrackedDeviceConnected(deviceIndex)) {
			continue;
		}
		printf("%d:(%.2f,%.2f,%.2f) ", deviceIndex, devicePos[deviceIndex].x, devicePos[deviceIndex].y, devicePos[deviceIndex].z);
	}
	printf("\n");
}


int app(int argc, const char** argv) {
	// Initialize stuff
	vr::EVRInitError error = vr::VRInitError_Compositor_Failed;
	std::cout << "Looking for SteamVR..." << std::flush;
	while (error != vr::VRInitError_None) {
		m_VRSystem = vr::VR_Init(&error, vr::VRApplication_Overlay);
		if (error != vr::VRInitError_None) {
			std::cout << "\nFailed due to reason " << VR_GetVRInitErrorAsSymbol(error) << "\n" << std::flush;
			std::cout << "Trying again in a few seconds...\n" << std::flush;
			std::this_thread::sleep_for(std::chrono::seconds(4));
		}
	}
	std::cout << "Success!\n";
	std::cout << "Looking for VR Input Emulator..." << std::flush;
	while (true) {
		try {
			inputEmulator.connect();
			break;
		}
		catch (vrinputemulator::vrinputemulator_connectionerror e) {
			std::cout << "\nFailed to connect to open vr input emulator, ensure you've installed it. If you have, try running this fix: https://drive.google.com/open?id=1Gn3IOm6GbkINplbEenu0zTr3DkB1E8Hc \n" << std::flush;
			std::this_thread::sleep_for(std::chrono::seconds(4));
			continue;
		}
	}
	std::cout << "Success!\n";

	// Create fake pelvis tracker
	pelvisTrackerID = createTracker();

#ifdef _WIN32
	if (SetConsoleCtrlHandler((PHANDLER_ROUTINE)ConsoleHandler, TRUE) == FALSE) {
		std::cerr << "Failed to set console handler, I won't shut down properly!\n";
	}
#endif
	signal(SIGINT, signalHandler);
	std::cout << "READY! Press CTRL+C with this window focused to quit.\n" << std::flush;


	bool running = true;
	// appliedImpulse = true;
	auto lastTime = std::chrono::high_resolution_clock::now();
	int numFramePresents = 0;
	while (running) {
		if (vr::VRCompositor() != NULL) {
			vr::Compositor_FrameTiming t;
			t.m_nSize = sizeof(vr::Compositor_FrameTiming);
			bool hasFrame = vr::VRCompositor()->GetFrameTiming(&t, 0);
			auto currentTime = std::chrono::high_resolution_clock::now();
			std::chrono::duration<float> dt = currentTime - lastTime;
			deltaTime = dt.count();
			// If the frame has changed we update, if a frame was redisplayed we update.
			if ((hasFrame && currentFrame != t.m_nFrameIndex) || (hasFrame && t.m_nNumFramePresents != numFramePresents)) {
				currentFrame = t.m_nFrameIndex;
				numFramePresents = t.m_nNumFramePresents;
				lastTime = currentTime;

				// Adjust Tracker Thingi
				Update();

				// Sleep for just under 1/90th of a second, so that maybe the next frame will be available.
				std::this_thread::sleep_for(std::chrono::microseconds(10000));
			}
			else {
				// Still waiting on the next frame, wait less this time.
				std::this_thread::sleep_for(std::chrono::microseconds(1111));
			}
		}
	}
	logFile.close();

	return 0;
}

int main(int argc, const char** argv) {
	return app(argc, argv);
}
