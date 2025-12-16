import React, { useState, useCallback } from "react";
import { Button, H1, Image, Text, View, XStack, YStack } from "tamagui";
import {
  Heart,
  MessageCircle,
  Share,
  Plus,
  Settings,
} from "@tamagui/lucide-icons";
import { useRouter, useFocusEffect } from "expo-router";
import {
  StatusBar,
  Share as RNShare,
  Platform,
  Alert,
  FlatList,
  RefreshControl,
  ActivityIndicator,
} from "react-native";
import { BlurView } from "expo-blur";
import { ResponsiveContainer } from "../../components/ResponsiveContainer";
import { ImageShimmer } from "../../components/image";
import { PostCard, PostCardShimmer } from "../../components/PostCard";
import { useFeed, FeedPost } from "../../hooks/useFeed";
import { LinearGradient } from "expo-linear-gradient";
import FleekLogo from "../../components/FleekLogo";
import {
  SafeAreaView,
  useSafeAreaInsets,
} from "react-native-safe-area-context";
import { ImagePost, Stats } from "../../types/shared";

const BlurViewForAndroid = () => {
  return (
    <View backgroundColor="rgba(17, 17, 17, 0.9)" width="100%" height="100%" />
  );
};

export default function FeedPage() {
  console.log("📱 FeedPage: Component starting...");

  const router = useRouter();
  const {
    posts,
    isLoading,
    isRefreshing,
    error,
    hasMore,
    refreshFeed,
    loadMore,
  } = useFeed();

  console.log("📱 FeedPage: State initialized", { posts, isLoading });

  const insets = useSafeAreaInsets();

  const renderPost = ({ item }: { item: FeedPost }) => <PostCard post={item} />;

  const renderFooter = () => {
    if (!hasMore && posts.length > 0) {
      return (
        <YStack padding="$4" alignItems="center">
          <Text color="#606060" fontSize="$3">
            No more posts
          </Text>
        </YStack>
      );
    }

    if (isLoading && posts.length > 0) {
      return (
        <YStack alignItems="center">
          <ActivityIndicator size="small" color="#4444dd" />
        </YStack>
      );
    }

    return null;
  };

  const renderEmpty = () => (
    <YStack flex={1} alignItems="center" justifyContent="center" padding="$6">
      <Text color="#808080" fontSize="$5" textAlign="center">
        {error ? `Connection Error` : "No posts yet"}
      </Text>
      <Text color="#606060" fontSize="$3" textAlign="center" marginTop="$2">
        {error
          ? "Could not load feed. Using offline mode."
          : "Be the first to create something amazing"}
      </Text>
      {error && (
        <Text color="#505050" fontSize="$2" textAlign="center" marginTop="$1">
          {error}
        </Text>
      )}

      <Button
        backgroundColor="#282828"
        marginTop="$6"
        paddingHorizontal="$6"
        paddingVertical="$3"
        borderRadius="$9"
        pressStyle={{ scale: 0.98, backgroundColor: "#333" }}
        onPress={error ? refreshFeed : () => router.push("/create")}
      >
        <Text color="white" fontSize="$4">
          {error ? "Retry Connection" : "Start Creating"}
        </Text>
      </Button>
    </YStack>
  );

  // Render shimmer loading for initial load
  const renderShimmerLoading = () => (
    <ResponsiveContainer backgroundColor="#111111">
      <StatusBar barStyle="light-content" backgroundColor="#111111" />

      {/* Content with padding for header */}
      <View position="relative" flex={1}>
        {/* Shimmer posts */}
        <FlatList
          data={[1, 2, 3, 4]} // Show 4 shimmer cards
          renderItem={() => <PostCardShimmer />}
          keyExtractor={(item) => `shimmer-${item}`}
          ItemSeparatorComponent={() => <View height={16} />}
          showsVerticalScrollIndicator={false}
          contentContainerStyle={{
            paddingTop: 80,
            paddingBottom: 20,
          }}
          style={{ flex: 1 }}
        />

        {/* Header (same as normal state) */}
        {Platform.OS === "ios" ? (
          <View position="absolute" top={0} left={0} right={0} zIndex={1000}>
            <BlurView
              intensity={80}
              tint="dark"
              style={{
                paddingHorizontal: 8,
                paddingTop: insets.top + 12,
                paddingBottom: 8,
              }}
            >
              <XStack justifyContent="space-between" alignItems="center">
                <XStack alignItems="center">
                  <FleekLogo width={32} height={32} />
                  <Text color="white" fontSize="$4" fontWeight="600">
                    Fleek
                  </Text>
                </XStack>
              </XStack>
            </BlurView>
          </View>
        ) : (
          <View position="absolute" top={0} left={0} right={0} zIndex={1000}>
            <XStack
              justifyContent="space-between"
              alignItems="center"
              marginTop={insets.top + 12}
              paddingHorizontal="$4"
              paddingVertical="$3"
              backgroundColor="transparent"
              zIndex={1001}
            >
              <XStack alignItems="center">
                <FleekLogo width={32} height={32} />
                <Text color="white" fontSize="$3" fontWeight="600">
                  Fleek
                </Text>
              </XStack>
            </XStack>
            <View
              position="absolute"
              top={0}
              left={0}
              right={0}
              bottom={0}
              zIndex={1000}
            >
              <BlurViewForAndroid />
            </View>
          </View>
        )}
      </View>
    </ResponsiveContainer>
  );

  if (isLoading && posts.length === 0) {
    return renderShimmerLoading();
  }

  console.log("📱 FeedPage: Posts", posts);

  return (
    <SafeAreaView
      style={{
        flex: 1,
        backgroundColor: "transparent",
      }}
      edges={["left", "right"]}
    >
      <StatusBar
        barStyle="light-content"
        backgroundColor="transparent"
        translucent
      />

      {/* Content with padding for header */}
      <View position="relative" flex={1}>
        {/* Feed with FlatList */}
        <FlatList
          data={posts}
          renderItem={renderPost}
          keyExtractor={(item, index) =>
            item.content_id
              ? `${item.content_id}-${index}`
              : `fallback-${index}-${Date.now()}`
          }
          ListEmptyComponent={renderEmpty}
          ItemSeparatorComponent={() => <View height={16} />}
          refreshControl={
            <RefreshControl
              refreshing={isRefreshing}
              onRefresh={refreshFeed}
              tintColor="#ffffff"
              titleColor="#ffffff"
              colors={["#4444dd", "#6666ff", "#8888ff"]}
              progressBackgroundColor="#333333"
            />
          }
          onEndReached={loadMore}
          onEndReachedThreshold={0.1}
          showsVerticalScrollIndicator={false}
          bounces={true}
          alwaysBounceVertical={true}
          contentContainerStyle={{
            paddingTop: insets.top + 32,
            ...(posts.length === 0 ? { flexGrow: 1 } : { paddingBottom: 20 }),
          }}
          style={{
            flex: 1,
            backgroundColor: "#111111",
          }}
          contentInsetAdjustmentBehavior="never"
        />

        {/* Floating Header with BlurView */}

        {Platform.OS === "ios" ? (
          <View position="absolute" top={0} left={0} right={0} zIndex={1000}>
            <BlurView
              intensity={80}
              tint="dark"
              style={{
                paddingHorizontal: 8,
                paddingTop: insets.top - 4,
                paddingBottom: 10,
                paddingVertical: 10,
              }}
            >
              <XStack
                justifyContent="space-between"
                alignItems="center"
                marginTop="$2"
              >
                <XStack alignItems="center">
                  <FleekLogo width={32} height={32} />
                  <Text color="white" fontSize="$4" fontWeight="600">
                    Fleek
                  </Text>
                </XStack>
              </XStack>
            </BlurView>
          </View>
        ) : (
          <View position="absolute" top={0} left={0} right={0} zIndex={1000}>
            <XStack
              justifyContent="space-between"
              alignItems="center"
              marginTop={insets.top + 4}
              paddingHorizontal="$4"
              paddingVertical="$2"
              backgroundColor="transparent"
              zIndex={1001}
            >
              <XStack alignItems="center">
                <FleekLogo width={32} height={32} />
                <Text color="white" fontSize="$3" fontWeight="600">
                  Fleek
                </Text>
              </XStack>
            </XStack>
            <View
              position="absolute"
              top={0}
              left={0}
              right={0}
              bottom={0}
              zIndex={1000}
            >
              <BlurViewForAndroid />
            </View>
          </View>
        )}
      </View>
    </SafeAreaView>
  );
}
