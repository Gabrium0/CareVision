import Foundation

/// The IPF1 binary frame header, byte-for-byte identical to `core/ipad_camera.py`
/// (`FRAME_HEADER = struct.Struct("<4sIdHHHH")`, 24 bytes, little-endian):
///
///   offset 0  : 4s  magic "IPF1"
///   offset 4  : I   seq          (uint32, wraps)
///   offset 8  : d   mediaTime    (float64 seconds)
///   offset 16 : H   width        (uint16)
///   offset 18 : H   height       (uint16)
///   offset 20 : H   chunk_index  (uint16)
///   offset 22 : H   chunk_count  (uint16)
///
/// then the JPEG (chunk) bytes. Over the LAN socket a whole JPEG fits in one
/// WebSocket binary message, so chunk_index/chunk_count are 0/1 for single-shot
/// frames; the header keeps the chunk fields so the same receiver code
/// (`_ingest_frame`) reassembles either transport.
enum FrameProtocol {
    static let magic: [UInt8] = Array("IPF1".utf8)   // 0x49 0x50 0x46 0x31
    static let headerSize = 24

    /// Build one 24-byte header. All multi-byte fields are little-endian.
    static func header(seq: UInt32, mediaTime: Double, width: UInt16, height: UInt16,
                       chunkIndex: UInt16 = 0, chunkCount: UInt16 = 1) -> Data {
        var data = Data(capacity: headerSize)
        data.append(contentsOf: magic)
        appendLE(&data, seq)
        appendLE(&data, mediaTime.bitPattern)          // f64 as its IEEE-754 bits
        appendLE(&data, width)
        appendLE(&data, height)
        appendLE(&data, chunkIndex)
        appendLE(&data, chunkCount)
        return data
    }

    /// One complete frame message = header + JPEG payload.
    static func frameMessage(seq: UInt32, mediaTime: Double, width: UInt16,
                             height: UInt16, jpeg: Data) -> Data {
        var msg = header(seq: seq, mediaTime: mediaTime, width: width, height: height)
        msg.append(jpeg)
        return msg
    }

    private static func appendLE<T: FixedWidthInteger>(_ data: inout Data, _ value: T) {
        var le = value.littleEndian
        withUnsafeBytes(of: &le) { data.append(contentsOf: $0) }
    }
}
