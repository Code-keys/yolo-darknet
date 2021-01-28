#ifndef TENSORRT_LOGGING_H
#define TENSORRT_LOGGING_H

#include "NvInferRuntimeCommon.h"
#include <cassert>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <sstream>
#include <string>

using Severity = nvinfer1::ILogger::Severity;

class LogStreamConsumerBuffer : public std::stringbuf
{
public:
    LogStreamConsumerBuffer(std::ostream& stream, const std::string& prefix, bool shouldLog)
        : mOutput(stream)
        , mPrefix(prefix)
        , mShouldLog(shouldLog)
    {
    }

    LogStreamConsumerBuffer(LogStreamConsumerBuffer&& other)
        : mOutput(other.mOutput)
    {
    }

    ~LogStreamConsumerBuffer()
    {
        if (pbase() != pptr())
        {
            putOutput();
        }
    }

    virtual int sync()
    {
        putOutput();
        return 0;
    }

    void putOutput()
    {
        if (mShouldLog)
        {
            std::time_t timestamp = std::time(nullptr);
            tm* tm_local = std::localtime(&timestamp);
            std::cout << "[";
            std::cout << std::setw(2) << std::setfill('0') << 1 + tm_local->tm_mon << "/";
            std::cout << std::setw(2) << std::setfill('0') << tm_local->tm_mday << "/";
            std::cout << std::setw(4) << std::setfill('0') << 1900 + tm_local->tm_year << "-";
            std::cout << std::setw(2) << std::setfill('0') << tm_local->tm_hour << ":";
            std::cout << std::setw(2) << std::setfill('0') << tm_local->tm_min << ":";
            std::cout << std::setw(2) << std::setfill('0') << tm_local->tm_sec << "] ";
            mOutput << mPrefix << str();
            str("");
            mOutput.flush();
        }
    }

    void setShouldLog(bool shouldLog)
    {
        mShouldLog = shouldLog;
    }

private:
    std::ostream& mOutput;
    std::string mPrefix;
    bool mShouldLog;
};

class LogStreamConsumerBase
{
public:
    LogStreamConsumerBase(std::ostream& stream, const std::string& prefix, bool shouldLog)
        : mBuffer(stream, prefix, shouldLog)
    {
    }

protected:
    LogStreamConsumerBuffer mBuffer;
};

class LogStreamConsumer : protected LogStreamConsumerBase, public std::ostream
{
public:
    LogStreamConsumer(Severity reportableSeverity, Severity severity)
        : LogStreamConsumerBase(severityOstream(severity), severityPrefix(severity), severity <= reportableSeverity)
        , std::ostream(&mBuffer) // links the stream buffer with the stream
        , mShouldLog(severity <= reportableSeverity)
        , mSeverity(severity)
    {
    }

    LogStreamConsumer(LogStreamConsumer&& other)
        : LogStreamConsumerBase(severityOstream(other.mSeverity), severityPrefix(other.mSeverity), other.mShouldLog)
        , std::ostream(&mBuffer) // links the stream buffer with the stream
        , mShouldLog(other.mShouldLog)
        , mSeverity(other.mSeverity)
    {
    }

    void setReportableSeverity(Severity reportableSeverity)
    {
        mShouldLog = mSeverity <= reportableSeverity;
        mBuffer.setShouldLog(mShouldLog);
    }

private:
    static std::ostream& severityOstream(Severity severity)
    {
        return severity >= Severity::kINFO ? std::cout : std::cerr;
    }

    static std::string severityPrefix(Severity severity)
    {
        switch (severity)
        {
        case Severity::kINTERNAL_ERROR: return "[F] ";
        case Severity::kERROR: return "[E] ";
        case Severity::kWARNING: return "[W] ";
        case Severity::kINFO: return "[I] ";
        case Severity::kVERBOSE: return "[V] ";
        default: assert(0); return "";
        }
    }

    bool mShouldLog;
    Severity mSeverity;
};

class Logger : public nvinfer1::ILogger
{
public:
    Logger(Severity severity = Severity::kWARNING)
        : mReportableSeverity(severity)
    {
    }

    enum class TestResult
    {
        kRUNNING, //!< The test is running
        kPASSED,  //!< The test passed
        kFAILED,  //!< The test failed
        kWAIVED   //!< The test was waived
    };

    nvinfer1::ILogger& getTRTLogger()
    {
        return *this;
    }

    void log(Severity severity, const char* msg) override
    {
        LogStreamConsumer(mReportableSeverity, severity) << "[TRT] " << std::string(msg) << std::endl;
    }

    void setReportableSeverity(Severity severity)
    {
        mReportableSeverity = severity;
    }

    class TestAtom
    {
    public:
        TestAtom(TestAtom&&) = default;

    private:
        friend class Logger;

        TestAtom(bool started, const std::string& name, const std::string& cmdline)
            : mStarted(started)
            , mName(name)
            , mCmdline(cmdline)
        {
        }

        bool mStarted;
        std::string mName;
        std::string mCmdline;
    };

    static TestAtom defineTest(const std::string& name, const std::string& cmdline)
    {
        return TestAtom(false, name, cmdline);
    }

    static TestAtom defineTest(const std::string& name, int argc, char const* const* argv)
    {
        auto cmdline = genCmdlineString(argc, argv);
        return defineTest(name, cmdline);
    }

    static void reportTestStart(TestAtom& testAtom)
    {
        reportTestResult(testAtom, TestResult::kRUNNING);
        assert(!testAtom.mStarted);
        testAtom.mStarted = true;
    }

    static void reportTestEnd(const TestAtom& testAtom, TestResult result)
    {
        assert(result != TestResult::kRUNNING);
        assert(testAtom.mStarted);
        reportTestResult(testAtom, result);
    }

    static int reportPass(const TestAtom& testAtom)
    {
        reportTestEnd(testAtom, TestResult::kPASSED);
        return EXIT_SUCCESS;
    }

    static int reportFail(const TestAtom& testAtom)
    {
        reportTestEnd(testAtom, TestResult::kFAILED);
        return EXIT_FAILURE;
    }

    static int reportWaive(const TestAtom& testAtom)
    {
        reportTestEnd(testAtom, TestResult::kWAIVED);
        return EXIT_SUCCESS;
    }

    static int reportTest(const TestAtom& testAtom, bool pass)
    {
        return pass ? reportPass(testAtom) : reportFail(testAtom);
    }

    Severity getReportableSeverity() const
    {
        return mReportableSeverity;
    }

private:
    static const char* severityPrefix(Severity severity)
    {
        switch (severity)
        {
        case Severity::kINTERNAL_ERROR: return "[F] ";
        case Severity::kERROR: return "[E] ";
        case Severity::kWARNING: return "[W] ";
        case Severity::kINFO: return "[I] ";
        case Severity::kVERBOSE: return "[V] ";
        default: assert(0); return "";
        }
    }

    static const char* testResultString(TestResult result)
    {
        switch (result)
        {
        case TestResult::kRUNNING: return "RUNNING";
        case TestResult::kPASSED: return "PASSED";
        case TestResult::kFAILED: return "FAILED";
        case TestResult::kWAIVED: return "WAIVED";
        default: assert(0); return "";
        }
    }

    static std::ostream& severityOstream(Severity severity)
    {
        return severity >= Severity::kINFO ? std::cout : std::cerr;
    }

    static void reportTestResult(const TestAtom& testAtom, TestResult result)
    {
        severityOstream(Severity::kINFO) << "&&&& " << testResultString(result) << " " << testAtom.mName << " # "
                                         << testAtom.mCmdline << std::endl;
    }

    static std::string genCmdlineString(int argc, char const* const* argv)
    {
        std::stringstream ss;
        for (int i = 0; i < argc; i++)
        {
            if (i > 0)
                ss << " ";
            ss << argv[i];
        }
        return ss.str();
    }

    Severity mReportableSeverity;
};

namespace
{
inline LogStreamConsumer LOG_VERBOSE(const Logger& logger)
{
    return LogStreamConsumer(logger.getReportableSeverity(), Severity::kVERBOSE);
}

inline LogStreamConsumer LOG_INFO(const Logger& logger)
{
    return LogStreamConsumer(logger.getReportableSeverity(), Severity::kINFO);
}

inline LogStreamConsumer LOG_WARN(const Logger& logger)
{
    return LogStreamConsumer(logger.getReportableSeverity(), Severity::kWARNING);
}

inline LogStreamConsumer LOG_ERROR(const Logger& logger)
{
    return LogStreamConsumer(logger.getReportableSeverity(), Severity::kERROR);
}

inline LogStreamConsumer LOG_FATAL(const Logger& logger)
{
    return LogStreamConsumer(logger.getReportableSeverity(), Severity::kINTERNAL_ERROR);
}

} // anonymous namespace

#endif // TENSORRT_LOGGING_H
