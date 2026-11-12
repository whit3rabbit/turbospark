import Foundation
import XCTest

@testable import TurboSparkApp

/// Task 6.12 (requirement 3.9): is an additional model scan folder reachable?
///
/// Every case runs against a temporary directory standing in for `/Volumes`
/// (`volumesRoot`), so no test reads or writes a real drive. A "mounted drive"
/// is a folder created under that fake root; unplugging it is never creating it.
final class ScanFolderStatusTests: XCTestCase {
    /// Holds the fake volumes root and anything outside it.
    private var base: URL!
    /// The fake `/Volumes`.
    private var root: URL!

    override func setUpWithError() throws {
        base = FileManager.default.temporaryDirectory
            .appendingPathComponent("scan-folder-status-\(UUID().uuidString)", isDirectory: true)
        root = base.appendingPathComponent("Volumes", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: base)
    }

    /// Creates `relative` under the fake volumes root and returns its path.
    @discardableResult
    private func mount(_ relative: String) throws -> String {
        let url = root.appendingPathComponent(relative, isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url.path
    }

    private func status(_ path: String) -> ScanFolderStatus {
        ModelStorageManager.scanFolderStatus(path, volumesRoot: root.path)
    }

    // MARK: - The three answers

    func testAFolderOnAMountedVolumeIsConnected() throws {
        let folder = try mount("T7/models")
        XCTAssertEqual(status(folder), .connected)
    }

    func testAFolderOnAVolumeThatIsNotMountedIsNotConnectedAndNamesTheVolume() {
        XCTAssertEqual(status(root.path + "/T7/models"), .notConnected(volumeName: "T7"))
        // Deeper than one level: the volume is the first component only.
        XCTAssertEqual(
            status(root.path + "/T7/TurboSpark/text/models"), .notConnected(volumeName: "T7"))
    }

    func testAMissingFolderOnAMountedVolumeIsNotFoundNotNotConnected() throws {
        try mount("T7")
        XCTAssertEqual(status(root.path + "/T7/gone"), .notFound)
        XCTAssertEqual(status(root.path + "/T7/gone/deeper"), .notFound)
    }

    // MARK: - Volume names that look alike

    func testAVolumeThatOnlySharesAPrefixIsNotTheSameVolume() throws {
        // T70 is plugged in; T7 is not. The path is on T7.
        try mount("T70/models")
        XCTAssertEqual(status(root.path + "/T7/models"), .notConnected(volumeName: "T7"))
    }

    func testAVolumeThatExtendsTheNameOfAMountedOneIsNotConnectedEither() throws {
        // The other way round: T7 is plugged in; the path is on T70.
        try mount("T7/models")
        XCTAssertEqual(status(root.path + "/T70/models"), .notConnected(volumeName: "T70"))
        XCTAssertEqual(status(root.path + "/T7/models"), .connected)
    }

    func testAVolumeNameWithSpacesAndDigitsIsOneComponent() throws {
        try mount("Samsung T7 2/models")
        XCTAssertEqual(status(root.path + "/Samsung T7 2/models"), .connected)
        XCTAssertEqual(
            status(root.path + "/Samsung T7/models"), .notConnected(volumeName: "Samsung T7"))
    }

    // MARK: - Spelling of the path

    func testATrailingSlashChangesNothing() throws {
        let folder = try mount("T7/models")
        XCTAssertEqual(status(folder + "/"), .connected)
        XCTAssertEqual(status(folder + "//"), .connected)
        XCTAssertEqual(status(root.path + "/Gone/models/"), .notConnected(volumeName: "Gone"))
        XCTAssertEqual(status(root.path + "/T7/missing/"), .notFound)
    }

    func testCurrentDirectoryComponentsAndRepeatedSlashesAreDropped() throws {
        try mount("T7/models")
        XCTAssertEqual(status(root.path + "/T7/./models"), .connected)
        XCTAssertEqual(status(root.path + "//T7//models"), .connected)
        XCTAssertEqual(status(root.path + "/./Gone/./models"), .notConnected(volumeName: "Gone"))
    }

    func testParentReferencesAreResolvedBeforeTheVolumeIsRead() throws {
        try mount("T7/models")
        // Back into the same volume.
        XCTAssertEqual(status(root.path + "/T7/../T7/models"), .connected)
        // The volume the path ENDS UP on decides, not the one it starts on.
        XCTAssertEqual(status(root.path + "/T7/../Gone/models"), .notConnected(volumeName: "Gone"))
        XCTAssertEqual(status(root.path + "/Gone/../T7/models"), .connected)
        XCTAssertEqual(status(root.path + "/T7/models/../../T7/models"), .connected)
    }

    func testParentReferencesThatLeaveTheVolumesRootAreOutsideIt() throws {
        let elsewhere = base.appendingPathComponent("elsewhere", isDirectory: true)
        try FileManager.default.createDirectory(at: elsewhere, withIntermediateDirectories: true)
        try mount("T7")
        XCTAssertEqual(status(root.path + "/T7/../../elsewhere"), .connected)
        XCTAssertEqual(status(root.path + "/T7/../../missing"), .notFound)
        // Past the filesystem root is the filesystem root, never a trap.
        XCTAssertEqual(status("/../../../" + "definitely/not/here/\(UUID().uuidString)"), .notFound)
    }

    func testTildeIsExpandedBeforeTheVolumeIsRead() {
        // Nothing here touches the disk: the fake root sits under the home
        // directory and `pathExists` answers from a table.
        let home = NSHomeDirectory()
        let fakeRoot = home + "/FakeVolumes"
        let mounted: Set<String> = [fakeRoot + "/T7", fakeRoot + "/T7/models"]
        let exists: (String) -> Bool = { mounted.contains($0) }

        XCTAssertEqual(
            ModelStorageManager.scanFolderStatus("~/FakeVolumes/T7/models", volumesRoot: fakeRoot, pathExists: exists),
            .connected)
        XCTAssertEqual(
            ModelStorageManager.scanFolderStatus("~/FakeVolumes/Gone/models", volumesRoot: fakeRoot, pathExists: exists),
            .notConnected(volumeName: "Gone"))
        // The root itself may be spelled with a tilde.
        XCTAssertEqual(
            ModelStorageManager.scanFolderStatus("~/FakeVolumes/Gone/models", volumesRoot: "~/FakeVolumes", pathExists: exists),
            .notConnected(volumeName: "Gone"))
    }

    func testSurroundingWhitespaceIsIgnoredLikeExpandPathDoes() throws {
        let folder = try mount("T7/models")
        XCTAssertEqual(status("  " + folder + "\n"), .connected)
    }

    func testAnEmptyOrBlankPathIsNotFound() {
        XCTAssertEqual(status(""), .notFound)
        XCTAssertEqual(status("   "), .notFound)
        XCTAssertEqual(status("\n\t"), .notFound)
    }

    // MARK: - Paths that are not on a volume

    func testAPathOutsideTheVolumesRootIsConnectedWhenItExistsAndNotFoundWhenItDoesNot() throws {
        let plain = base.appendingPathComponent("plain/models", isDirectory: true)
        try FileManager.default.createDirectory(at: plain, withIntermediateDirectories: true)
        XCTAssertEqual(status(plain.path), .connected)
        XCTAssertEqual(status(base.path + "/plain/absent"), .notFound)
    }

    func testARootThatOnlySharesAPrefixWithTheVolumesRootIsNotUnderIt() throws {
        // `<base>/Volumes2/T7/models` is not on a volume of `<base>/Volumes`.
        // A string prefix would read it as volume "2" or as volume "T7".
        let sibling = base.appendingPathComponent("Volumes2/T7/models", isDirectory: true)
        XCTAssertEqual(status(sibling.path), .notFound)
        try FileManager.default.createDirectory(at: sibling, withIntermediateDirectories: true)
        XCTAssertEqual(status(sibling.path), .connected)
    }

    func testTheVolumesRootItselfIsNotAVolume() {
        XCTAssertEqual(status(root.path), .connected)
        XCTAssertEqual(status(root.path + "/"), .connected)
    }

    // MARK: - What counts as a mount point

    func testAMountPointThatIsASymlinkCountsAsMounted() throws {
        // The boot volume appears as `/Volumes/Macintosh HD`, a link to `/`.
        let boot = base.appendingPathComponent("boot/Users/me/models", isDirectory: true)
        try FileManager.default.createDirectory(at: boot, withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(
            atPath: root.path + "/Macintosh HD", withDestinationPath: base.path + "/boot")

        XCTAssertEqual(status(root.path + "/Macintosh HD/Users/me/models"), .connected)
        // The link is the mount point: a missing folder behind it is not an
        // unplugged drive.
        XCTAssertEqual(status(root.path + "/Macintosh HD/Users/me/absent"), .notFound)
    }

    func testALinkThatPointsNowhereIsNotAMountPoint() throws {
        try FileManager.default.createSymbolicLink(
            atPath: root.path + "/Dangling", withDestinationPath: base.path + "/does-not-exist")
        XCTAssertEqual(status(root.path + "/Dangling/models"), .notConnected(volumeName: "Dangling"))
    }

    func testOnlyTheMountPointIsAskedAboutWhenTheVolumeIsGone() {
        var asked: [String] = []
        let result = ModelStorageManager.scanFolderStatus(
            "/Volumes/T7/TurboSpark/text", volumesRoot: "/Volumes",
            pathExists: { asked.append($0); return false })
        XCTAssertEqual(result, .notConnected(volumeName: "T7"))
        // The dead path is never touched: one question, about the mount point.
        XCTAssertEqual(asked, ["/Volumes/T7"])
    }

    func testTheFolderIsAskedAboutOnlyAfterTheMountPointExists() {
        var asked: [String] = []
        let result = ModelStorageManager.scanFolderStatus(
            "/Volumes/T7/TurboSpark/text", volumesRoot: "/Volumes",
            pathExists: { asked.append($0); return $0 == "/Volumes/T7" })
        XCTAssertEqual(result, .notFound)
        XCTAssertEqual(asked, ["/Volumes/T7", "/Volumes/T7/TurboSpark/text"])
    }

    func testTheDefaultRootIsVolumes() {
        // `pathExists` answers, so no real volume is read.
        XCTAssertEqual(
            ModelStorageManager.scanFolderStatus("/Volumes/T7/models", pathExists: { _ in false }),
            .notConnected(volumeName: "T7"))
        XCTAssertEqual(
            ModelStorageManager.scanFolderStatus("/Volumes/T7/models", pathExists: { _ in true }),
            .connected)
        // Not under /Volumes: the volume check does not apply at all.
        XCTAssertEqual(
            ModelStorageManager.scanFolderStatus("/Users/me/models", pathExists: { $0 == "/Users/me/models" }),
            .connected)
    }

    func testAVolumeNameComesBackExactlyAsTheDriveSpelledIt() {
        // Escaping is the display layer's job (`ScanFolderRowModel`): the
        // status carries the real name so a comparison is still exact.
        let hostile = "T7\u{202E}gpj"
        XCTAssertEqual(
            status(root.path + "/" + hostile + "/models"), .notConnected(volumeName: hostile))
    }
}
